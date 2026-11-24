$ErrorActionPreference = 'Stop'

$repositoryRoot = Split-Path -Parent $PSScriptRoot
$agentsPath = Join-Path $repositoryRoot 'AGENTS.md'
$expected = @(
    '注意事项'
    '·每次改动完成后，都必须创建一个对应的Gitcommit，以便后续追踪和回滚。'
    '·每次改动后，都必须编写或更新相关测试，并在交付给用户前，确保所有测试和验证全部通过。'
)

if (-not (Test-Path -LiteralPath $agentsPath -PathType Leaf)) {
    throw 'AGENTS.md is missing.'
}

$actual = @(Get-Content -LiteralPath $agentsPath -Encoding UTF8)
if ($actual.Count -ne $expected.Count) {
    throw 'AGENTS.md has an unexpected number of lines.'
}

for ($index = 0; $index -lt $expected.Count; $index++) {
    if ($actual[$index] -cne $expected[$index]) {
        throw "AGENTS.md content mismatch at line $($index + 1)."
    }
}

Write-Output 'PASS: AGENTS.md contains all requested instructions.'
