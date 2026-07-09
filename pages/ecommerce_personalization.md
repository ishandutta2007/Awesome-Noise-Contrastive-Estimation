# Open-Vocabulary Zero-Shot E-Commerce

[<- Back to Home](../README.md)

## Overview
Transforms consumer search architecture by projecting raw item images and conversational text into the same multi-dimensional coordinate space. Enables instant, highly accurate visual matching without requiring human data labeling.

## Architecture Architecture
```mermaid
graph TD;
    A[Product Photo] --> B[Vision Encoder];
    C[Search Query Text] --> D[Text Encoder];
    B --> E[Similarity Search];
    D --> E;
```
