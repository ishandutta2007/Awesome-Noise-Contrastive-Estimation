# Representation Collapse Deficit

[<- Back to Home](../README.md)

## Overview
Without explicit negative samples, neural networks naturally default to outputting a constant, identical vector for all inputs to trivially achieve zero contrastive loss. Solutions involve asymmetric architectures with Stop-Gradient tracking (BYOL) or variance preservation metrics (VICReg).

## Architecture Architecture
```mermaid
graph LR;
    A[View 1] --> C[Encoder];
    B[View 2] --> C;
    C --> D[Same Constant Vector];
    D --> E[Zero Loss / Collapse];
```
