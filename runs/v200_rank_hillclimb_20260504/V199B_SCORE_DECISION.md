# V199B Score Decision

Data: 2026-05-04

Submissao:

- Ref: `52325494`
- Descricao: `V199B exact-sha final candidate sha 19fa057a local-audit`
- Status: `SubmissionStatus.COMPLETE`
- Public score: `0.86`
- ZIP SHA: `19fa057ad55c569c490650b430b76809cacc18fd5f5fbb8f6c5bf8f65785e59c`
- Adapter SHA: `444dd40c44ac5eb8a642ebe00a5616ba41548657dbd01e17f3db8bda7e785c7a`

Decisao:

- Nao houve regressao de score publico versus baseline `0.86`.
- Nao houve melhoria de score publico.
- V199B nao deve ser promovido automaticamente.
- V194/ref `52275052` continua baseline de producao ate confirmacao de ranking/empate melhor ou score `> 0.86`.

Impacto para V200:

- V200A/V200B devem partir de V194.
- V199B pode ser usado como delta experimental em V200B, mas nao como novo baseline.
