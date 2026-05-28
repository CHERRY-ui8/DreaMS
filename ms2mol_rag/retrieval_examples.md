# DreaMS Retrieval Examples
Generated from 18362 validation samples, retrieved from 313613 training samples.
| # | Category | Tanimoto | Query SMILES | Neighbor SMILES | Shared Functional Groups |
|---|---|---|---|---|---|
| 1 | bottom (t=0.000) | 0.000 | `CC(=O)NC1(C(=O)O)CCC(C)CC1` | `c1ccc(CN(CCN(Cc2ccccn2)Cc2ccccn2)Cc2ccccn2)nc1` | (none) |
| 2 | bottom (t=0.000) | 0.000 | `O=CC=CC=CCCCCCCCC(=O)O` | `Cc1cccn2cc(-c3ccc(NC(C)c4cc(C)n(C)n4)cc3)nc12` | (none) |
| 3 | low (t=0.106) | 0.106 | `O=C(NC1CCCN(C(=O)c2ccccc2)C1)c1n[nH]c2c1CCCC2` | `CCN(CC)C(=O)c1ccc(O)cc1` | Amide (CONH), Aromatic Ring, Benzene Ring, Lactam |
| 4 | low (t=0.106) | 0.106 | `Cc1cc(C)c(N2CCN(S(=O)(=O)N(C)C)CC2)c(C)c1` | `c1cnc(OCC2CCCN(c3ccnc(C4CC4)n3)C2)nc1` | Aromatic Ring |
| 5 | mid (t=0.147) | 0.147 | `CC=C1C(OC2OC(CO)C(O)C(O)C2O)OC=C(C(=O)OC)C1CC(=O)OCC1OC(OCCc2ccc(OC(=O)CC3C(C(=O)OC)=COC(OC4OC(CO)C(O)C(O)C4O)C3=CC)cc2)C(O)C(O)C1O` | `CCOC(=O)C(=O)CC(=O)c1cc(OC)ccc1OC` | Aromatic Ring, Benzene Ring, Ester (COOR), Ether (ROR), Lactone |
| 6 | mid (t=0.147) | 0.147 | `COCCCN1CCc2[nH]c(SCc3cccc(Br)c3)nc(=O)c2C1` | `COc1ccc(Nc2c(-c3ccccc3O)nc3cnc(Br)cn23)cc1OC` | Aromatic Ring, Benzene Ring, Ether (ROR) |
| 7 | high (t=0.215) | 0.215 | `Cc1cc(Nc2ncc(Cl)c(Nc3ccccc3S(=O)(=O)C(C)C)n2)c(OC(C)C)cc1C1CCN(CC(=O)NCCNc2cccc3c2C(=O)N(C2CCC(=O)NC2=O)C3=O)CC1` | `C=CC(=O)Nc1cccc(Nc2nc(Nc3ccc(NC4CN(CCF)C4)cc3OC)ncc2C(F)(F)F)c1` | Amide (CONH), Aromatic Ring, Benzene Ring, Ether (ROR), Lactam, Pyrimidine |
| 8 | high (t=0.215) | 0.215 | `Cc1ccc(C)c(C(NC(=O)c2ccc(F)cc2)c2ccccc2)c1` | `CC(=CC(=O)Nc1ccccc1C(=O)O)c1ccc2ccccc2c1` | Amide (CONH), Aromatic Ring, Benzene Ring, Lactam |
| 9 | top (t=1.000) | 1.000 | `CCCCCCCCCCC=CCC=CCCC(=O)OC(COC(=O)CCCCCCCCCCCCCCCC)COP(=O)([O-])OCC[N+](C)(C)C` | `CCCCCCC=CCC=CCC=CCC=CCCC(=O)OC(COC(=O)CCCCCCCCCCCCCCCC)COP(=O)([O-])OCC[N+](C)(C)C` | Alkene (C=C), Ester (COOR), Ether (ROR), Lactone |
| 10 | top (t=1.000) | 1.000 | `CCC=CCC(O)C=CC=CCC=CCC=CCC=CCCC(=O)O` | `CCC=CCC=CCC=CCC=CC=CC(O)CC=CCCC(=O)O` | Alcohol (OH), Alkene (C=C), Carboxylic Acid (COOH) |

## Detailed Analysis
### Example 1: BOTTOM (Tanimoto=0.000)
- **Query (bottom)**: `CC(=O)NC1(C(=O)O)CCC(C)CC1`
- **Neighbor**: `c1ccc(CN(CCN(Cc2ccccn2)Cc2ccccn2)Cc2ccccn2)nc1`
- **Shared groups**: *none*
- **Query only groups**: Alcohol (OH), Amide (CONH), Carboxylic Acid (COOH), Lactam
- **Neighbor only groups**: Aromatic Ring, Pyridine
- **No direct substructure match** — different molecular scaffolds

---
### Example 2: BOTTOM (Tanimoto=0.000)
- **Query (bottom)**: `O=CC=CC=CCCCCCCCC(=O)O`
- **Neighbor**: `Cc1cccn2cc(-c3ccc(NC(C)c4cc(C)n(C)n4)cc3)nc12`
- **Shared groups**: *none*
- **Query only groups**: Alcohol (OH), Aldehyde (CHO), Alkene (C=C), Carboxylic Acid (COOH)
- **Neighbor only groups**: Aromatic Ring, Benzene Ring, Imidazole, Pyridine
- **No direct substructure match** — different molecular scaffolds

---
### Example 3: LOW (Tanimoto=0.106)
- **Query (low)**: `O=C(NC1CCCN(C(=O)c2ccccc2)C1)c1n[nH]c2c1CCCC2`
- **Neighbor**: `CCN(CC)C(=O)c1ccc(O)cc1`
- **Shared groups**: Amide (CONH), Aromatic Ring, Benzene Ring, Lactam
- **Neighbor only groups**: Alcohol (OH), Phenol (ArOH)
- **No direct substructure match** — different molecular scaffolds

---
### Example 4: LOW (Tanimoto=0.106)
- **Query (low)**: `Cc1cc(C)c(N2CCN(S(=O)(=O)N(C)C)CC2)c(C)c1`
- **Neighbor**: `c1cnc(OCC2CCCN(c3ccnc(C4CC4)n3)C2)nc1`
- **Shared groups**: Aromatic Ring
- **Query only groups**: Benzene Ring
- **Neighbor only groups**: Ether (ROR), Pyrimidine
- **No direct substructure match** — different molecular scaffolds

---
### Example 5: MID (Tanimoto=0.147)
- **Query (mid)**: `CC=C1C(OC2OC(CO)C(O)C(O)C2O)OC=C(C(=O)OC)C1CC(=O)OCC1OC(OCCc2ccc(OC(=O)CC3C(C(=O)OC)=COC(OC4OC(CO)C(O)C(O)C4O)C3=CC)cc2)C(O)C(O)C1O`
- **Neighbor**: `CCOC(=O)C(=O)CC(=O)c1cc(OC)ccc1OC`
- **Shared groups**: Aromatic Ring, Benzene Ring, Ester (COOR), Ether (ROR), Lactone
- **Query only groups**: Alcohol (OH), Alkene (C=C)
- **Neighbor only groups**: Ketone (C=O)
- **No direct substructure match** — different molecular scaffolds

---
### Example 6: MID (Tanimoto=0.147)
- **Query (mid)**: `COCCCN1CCc2[nH]c(SCc3cccc(Br)c3)nc(=O)c2C1`
- **Neighbor**: `COc1ccc(Nc2c(-c3ccccc3O)nc3cnc(Br)cn23)cc1OC`
- **Shared groups**: Aromatic Ring, Benzene Ring, Ether (ROR)
- **Query only groups**: Pyrimidine, Sulfide (RSR)
- **Neighbor only groups**: Alcohol (OH), Imidazole, Phenol (ArOH)
- **No direct substructure match** — different molecular scaffolds

---
### Example 7: HIGH (Tanimoto=0.215)
- **Query (high)**: `Cc1cc(Nc2ncc(Cl)c(Nc3ccccc3S(=O)(=O)C(C)C)n2)c(OC(C)C)cc1C1CCN(CC(=O)NCCNc2cccc3c2C(=O)N(C2CCC(=O)NC2=O)C3=O)CC1`
- **Neighbor**: `C=CC(=O)Nc1cccc(Nc2nc(Nc3ccc(NC4CN(CCF)C4)cc3OC)ncc2C(F)(F)F)c1`
- **Shared groups**: Amide (CONH), Aromatic Ring, Benzene Ring, Ether (ROR), Lactam, Pyrimidine
- **Neighbor only groups**: Alkene (C=C)
- **No direct substructure match** — different molecular scaffolds

---
### Example 8: HIGH (Tanimoto=0.215)
- **Query (high)**: `Cc1ccc(C)c(C(NC(=O)c2ccc(F)cc2)c2ccccc2)c1`
- **Neighbor**: `CC(=CC(=O)Nc1ccccc1C(=O)O)c1ccc2ccccc2c1`
- **Shared groups**: Amide (CONH), Aromatic Ring, Benzene Ring, Lactam
- **Neighbor only groups**: Alcohol (OH), Alkene (C=C), Carboxylic Acid (COOH)
- **No direct substructure match** — different molecular scaffolds

---
### Example 9: TOP (Tanimoto=1.000)
- **Query (top)**: `CCCCCCCCCCC=CCC=CCCC(=O)OC(COC(=O)CCCCCCCCCCCCCCCC)COP(=O)([O-])OCC[N+](C)(C)C`
- **Neighbor**: `CCCCCCC=CCC=CCC=CCC=CCCC(=O)OC(COC(=O)CCCCCCCCCCCCCCCC)COP(=O)([O-])OCC[N+](C)(C)C`
- **Shared groups**: Alkene (C=C), Ester (COOR), Ether (ROR), Lactone
- **No direct substructure match** — different molecular scaffolds

---
### Example 10: TOP (Tanimoto=1.000)
- **Query (top)**: `CCC=CCC(O)C=CC=CCC=CCC=CCC=CCCC(=O)O`
- **Neighbor**: `CCC=CCC=CCC=CCC=CC=CC(O)CC=CCCC(=O)O`
- **Shared groups**: Alcohol (OH), Alkene (C=C), Carboxylic Acid (COOH)
- **No direct substructure match** — different molecular scaffolds

---

## Summary
- **Total queries**: 18362
- **Mean Top-1 Tanimoto**: 0.2266
- **Median Top-1 Tanimoto**: 0.1566
- **% with Tanimoto > 0.4**: 14.9%
- **% with Tanimoto > 0.7**: 4.1%

### Verdict
DreaMS embeddings encode MS fragmentation patterns, not molecular structure. Retrieved neighbors tend to share **ionization behavior** (similar functional groups that fragment similarly in MS/MS) rather than **overall molecular topology**. This is expected for MS-based embeddings and explains the low Tanimoto scores.
