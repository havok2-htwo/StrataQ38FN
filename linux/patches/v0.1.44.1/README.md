# Our series on v0.1.44.1 (g3-1441)

Base: upstream tag v0.1.44.1 (`296b59d2`). 58 commits, branch g3-1441 at `f17f3eec` = the production engine
`strata-g3-1441a` (built in Strata-1441/build-1441 with CUDA 13.0). The 0.1.43 series (`../v0.1.43/`) rebased, plus
the commits made on g3-143 after it. The low-level levers in `../v0.1.44.1-ll/` and `../v0.1.44.1-ll2/` apply on top.

    git -C Strata worktree add ../Strata-1441 -b g3-1441 v0.1.44.1
    git -C Strata-1441 am ../linux/patches/v0.1.44.1/*.patch
