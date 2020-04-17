# NSSC 2 - Exercise 1

WSL settings tweaked:

```bash
sudo vim /etc/sysctl.d/10-ptrace.conf
```

## Task 1: Questions

[Info on Cache organization](https://en.wikichip.org/wiki/intel/xeon_gold/6248) --> Cascade Lake

L2: 20MiB (MebiBytes) pro CPU

1 Node = 2x20MiB

[Cascade Lake](https://en.wikichip.org/wiki/intel/microarchitectures/cascade_lake#Memory_Hierarchy)

L2 Cache:

    - 1 MiB/core, 16-way set associative
    - 64 B line size
    - Inclusive
    - 64 B/cycle bandwidth to L1$
    - Write-back policy
    - 14 cycles latency
