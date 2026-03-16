# CLAUDE.md

## Goal

Extend the existing **SystemC TLM-2.0 performance simulator** to model **partial-result accumulation in multi-threaded GEMM**.

The current simulator models:

- Workers
- Matrix accelerator
- Vector accelerator
- Memory
- Cycle-based timing
- TLM communication

The new feature must add **dependency-aware accumulation of partial results**.

---

## Problem

In multi-threaded GEMM:

C = A × B

When work is split across threads, each thread produces a **partial result**.

Example (split along K):

Worker 0: A[:,0:k0] × B[0:k0,:]  
Worker 1: A[:,k0:k1] × B[k0:k1,:]

These produce **partial C matrices**.

The final result requires **accumulation of all partial results**.


## New Behavior

The simulator must model two stages:

### 1. Multiplication
Workers send requests to accelerators and perform matrix multiplication of their own tiles

Each worker produces a **partial result**.

### 2. Accumulation
When any two partial results are ready:

- accmulate two partial results together
- execute it on **VectorAccelerator**
- keep accmulating until all partial results are accumulated

Accumulation **depends on multiplication completion**.

---

## Dependency Rule

Accumulation can start **only when all partial results are ready**.

## Requirements
- Use the existing simulator modules. 
- The accumulation of partial results can be in parallel if there are more than one pair of partial results ready to be sumed
- The simulator should be generalized for accelerator amounts

# Varification
- Create test cases and test the correctness of the implementation
