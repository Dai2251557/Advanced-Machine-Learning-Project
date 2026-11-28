"""冻结三策略测试终稿的 v3 盲评：整批封存后，每成功稿一次，不重试、不续跑。

--check 按冻结协议重建盲评材料，不需要 Key、金标或网络。--batch-dir 可指定只读归档；
--dispatch 仅接受 outputs/course-final 的执行封存，并独占新建其 judge/ 目录。
"""
import argparse
import json
import os
import urllib.error
import urllib.request
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path

from evaluation.judge import (JUDGE_MODEL, PROMPT_ID, build_judge_request, finalize_judge_result,
                              prepare_sealed_test_reviews, save_judge_result)
from utils.data_release import load_manifest, read_json

from . import BATCH, POLICIES, ROOT, new_output
from .run_final import preflight, save

JUDGE = BATCH / 'judge'
URL = 'https://api.deepseek.com/chat/completions'
WORKERS = 10


def inputs(batch):
    execution = batch / 'execution'
    manifest_path = execution / 'execution_complete.json'
    if not manifest_path.is_file():
        raise FileNotFoundError(f'需要完整原始封存（不是公开 CSV）：{manifest_path}')
    tasks, corpus, _, _, header = preflight()
    seal = read_json(manifest_path)
    if any(seal.get(k) != v for k, v in header.items()):
        raise ValueError('不是本轮冻结策略和测试环境对应的完整执行批次')
    jobs = prepare_sealed_test_reviews(execution_dir=execution, public_tasks=[t.to_dict() for t in tasks],
                                       frozen_papers=corpus['papers'], manifest=load_manifest('test'))
    bytid = {r['trajectory_id']: r for r in seal['episodes']}
    wanted = {k for k, r in bytid.items() if r['success']}
    if (len(bytid) != 30 or {(r['review_id'], r['policy']) for r in bytid.values()} !=
            {(t.review_id, p) for t in tasks for p in POLICIES} or
            len(jobs) != len(wanted) or {j['binding']['trajectory_id'] for j in jobs} != wanted):
        raise ValueError('完整批次与成功终稿集合不匹配')
    result = []
    for job in jobs:
        b = job['binding']  # Only the request body goes to the remote Judge, never this binding or row.
        row = bytid[b['trajectory_id']]
        if b['final_text_sha256'] != row['final_text_sha256'] or b['review_id'] != row['review_id'] or b['split'] != 'test':
            raise ValueError('评分身份不符')
        result.append((row, job, build_judge_request(job)))
    return result


def send(folder, row, prepared, body, key):
    from litreview.clients import NoRedirect
    req = urllib.request.Request(URL, data=json.dumps(body, ensure_ascii=False).encode(), method='POST',
                                 headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'})
    is_api_error = False
    try:
        with urllib.request.build_opener(NoRedirect()).open(req, timeout=180) as response:
            row['http_status'] = response.status
            data = json.load(response)
    except urllib.error.HTTPError as exc:
        row.update(status='http_error', http_status=exc.code)
        is_api_error = True
    except (urllib.error.URLError, TimeoutError, OSError):
        row['status'] = 'transport_unknown'
        is_api_error = True
    except (ValueError, UnicodeError):
        row['status'] = 'invalid_response_json'
        is_api_error = True
    else:
        receipt = {'binding': prepared['binding'], 'request': body, 'response': data}
        save(folder / 'receipt.json', receipt)
        row['usage'] = data.get('usage') if isinstance(data, dict) else None
        try:
            label = finalize_judge_result(prepared, receipt)
            row['label'] = str(save_judge_result(label, JUDGE / 'labels').relative_to(ROOT))
            row['status'] = 'labeled'
        except (ValueError, TypeError, KeyError, AttributeError, IndexError) as exc:
            row.update(status='invalid_receipt', error_type=type(exc).__name__)
        if not isinstance(data, dict) or data.get('model') != JUDGE_MODEL:
            is_api_error = True
    save(folder / 'result.json', row)
    return row, is_api_error


def dispatch():
    new_output(JUDGE)
    jobs = inputs(BATCH)
    from litreview.clients import DeepSeekBackend
    client = DeepSeekBackend.from_env()
    if client.base_url.rstrip('/') not in ('https://api.deepseek.com', 'https://api.deepseek.com/v1'):
        raise ValueError('非官方端点')
    JUDGE.mkdir(exist_ok=False)  # Final exclusive guard before the first physical request.
    rows = []
    streak = sent = 0
    stop = None
    pending = set()
    with (JUDGE / 'dispatch.jsonl').open('x', encoding='utf-8') as ledger, ThreadPoolExecutor(max_workers=WORKERS) as pool:
        while pending or (sent < len(jobs) and not stop):
            while not stop and sent < len(jobs) and len(pending) < WORKERS and not any(f.done() for f in pending):
                if sent >= 30:
                    raise RuntimeError('超出本批成功终稿上限')
                meta, prepared, body = jobs[sent]
                i = sent + 1
                folder = JUDGE / f'{i:02d}'
                folder.mkdir()
                save(folder / 'request.json', body)
                save(folder / 'binding.json', prepared['binding'])
                row = {k: meta[k] for k in ('review_id', 'policy', 'trajectory_id', 'final_text_sha256')}
                row.update(split='test', attempt=i, status='unknown', usage=None)
                ledger.write(json.dumps({'attempt': i, 'trajectory_id': row['trajectory_id'],
                                         'started_at': datetime.now(timezone.utc).isoformat()}) + '\n')
                ledger.flush()
                os.fsync(ledger.fileno())
                pending.add(pool.submit(send, folder, row, prepared, body, client.api_key))
                sent = i
            if pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    row, api_error = future.result()
                    rows.append(row)
                    streak = streak + 1 if api_error else 0
                    if streak >= 2:
                        stop = 'two consecutive Judge API errors'
                    print(len(rows), '/', len(jobs), row['review_id'], row['policy'], row['status'], flush=True)
    rows.sort(key=lambda r: r['attempt'])
    usage = {}
    for key in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
        vals = [(r['usage'] or {}).get(key) for r in rows]
        known = [n for n in vals if type(n) is int and n >= 0]
        usage[key] = {'known_sum': sum(known), 'unknown_calls': len(rows) - len(known)}
    summary = {'model': JUDGE_MODEL, 'prompt_id': PROMPT_ID, 'split': 'test', 'queue_count': len(jobs),
               'physical_dispatch_attempts': sent, 'successful_new_labels': sum(r['status'] == 'labeled' for r in rows),
               'not_sent': len(jobs) - sent, 'stop': stop, 'usage': usage, 'results': rows}
    save(JUDGE / 'summary.json', summary)
    print('J', sent, '有效标签', summary['successful_new_labels'], '未发送', summary['not_sent'], '用量', usage, flush=True)
    if stop:
        raise SystemExit(stop)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--check', action='store_true', help='仅读封存并准备盲评请求，不发送')
    group.add_argument('--dispatch', action='store_true', help='对本地固定批次发起计费的正式盲评，仅限单独授权')
    parser.add_argument('--batch-dir', type=Path, help='仅 --check：包含 execution/ 的只读封存批次；默认 outputs/course-final')
    args = parser.parse_args()
    if args.check:
        jobs = inputs(args.batch_dir or BATCH)
        print('30 条终态封存，可盲评成功稿', len(jobs), '；本次未发送请求')
    elif args.batch_dir is not None:
        parser.error('--batch-dir 只允许与 --check 一起使用，不能对归档输入重新 dispatch')
    else:
        dispatch()


if __name__ == '__main__':
    main()
