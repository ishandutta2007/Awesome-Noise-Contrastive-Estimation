# Unsupervised Biomolecular Sequence Alignment

[<- Back to Home](../README.md)

## Overview
Using contrastive objective functions to automatically group and map unannotated DNA, RNA, or peptide structures based on structural similarities. Critical for zero-shot mutation tracking and target drug discovery.

## Architecture Architecture
```mermaid
graph TD;
    A[Protein Sequence A] --> C[Siamese Network];
    B[Protein Sequence B] --> C;
    C --> D[Geometric Distance Metric];
```
