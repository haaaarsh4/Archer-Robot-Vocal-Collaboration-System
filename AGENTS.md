# Repository rules

## Commits
- Commit messages carry **no attribution of any kind**: no `Co-Authored-By`, no
  "Generated with ..." line, no tool or assistant names, no emoji trailers.
- Author and committer are always `haaaarsh4 <harshupadhyay2004@gmail.com>`.
- Never push, force-push, or touch a remote unless that exact request is in the
  current message.

## Files
- `.freebuff/` stays untracked and gitignored; it is local tool state, not code.
- Model weights, index files and caches stay gitignored. Do not add files over
  100 MB; GitHub rejects them.
