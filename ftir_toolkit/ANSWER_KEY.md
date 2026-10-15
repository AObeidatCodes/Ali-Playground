# Answer key: blind FTIR exercises

Do the exercises first (`exercise_sheet.py` or the sheets in your prep doc). For each spectrum,
write down: the main bands and their assignment, your identification, how sure you are, and the
next step.

Wavenumbers below are what the toolkit's peak picker reports. Literature values are given where
the measured band sits somewhere else, and the reason.

## Set A: NIST gas-phase spectra (band assignment)

Gas spectra show sharp rotational lines. In a liquid or a dried residue the same bands are
broader, and O-H moves from a sharp "free" band near 3650 to a broad hydrogen-bonded band
near 3200-3500.

| # | Answer | Key bands (cm-1) | What it teaches | Fab context |
| --- | --- | --- | --- | --- |
| A1 | Acetone | C=O 1739 (liquid about 1715); CH3 bend 1366; C-C-C 1218; no O-H, no N-H | A strong carbonyl with no O-H points to a ketone or ester | Cleaning solvent |
| A2 | Water vapour | A comb of sharp lines at 3500-3950 and 1300-1900, nothing else | An **artefact**: purge the instrument or take a fresh background before you interpret anything | Every lab, every day |
| A3 | Sulfur hexafluoride (SF6) | One very strong band at 948 (S-F); 615; **no C-H at 2850-3000** | No C-H means not an ordinary organic | Plasma etch and chamber-clean gas. FTIR on gases is used for exhaust monitoring, not particle analysis. |
| A4 | Isopropanol (IPA) | Sharp free O-H 3636; C-H 2982/2971; CH3 bends 1473/1382; C-O 1151/1072; 957 | Alcohol = O-H + C-O. As a residue the O-H would be broad near 3350. | The most common rinse and drying solvent |
| A5 | Carbon dioxide | Asymmetric stretch 2324-2360; bend at 670; combination bands 3717/3596 | An **artefact**. In B4 it shows up as a *negative* band. | Every lab, every day |

The library search returns "no reliable match" for all five. That is correct: the library holds
polymers, not gases. Lesson: a search is only as good as the library behind it.

## Set B: real measurements (library search + judgement)

| # | Answer | Library result | What it teaches |
| --- | --- | --- | --- |
| B1 | **Cellulose acetate**. One pixel from a real FTIR microscope map (25 um pixels, Open Specy `CA_tiny_map`). | Cellulose acetate r = 0.91, next 0.37: **probable match** | The clean case. Confirm it: broad O-H 3495, C-H 2955, ester C=O 1759, acetyl CH3 1381, C-O 1273 and 1103 (literature about 1235 and 1050; map spectra are often shifted). A large gap to the second hit makes the call strong. |
| B2 | **Polyethylene** with soil. ATR-type spectrum (Agilent .asp). | Top hit **polystyrene** r = 0.57; LDPE only 3rd (0.34): **weak** | **The library is wrong and your band assignment is right.** Four bands are enough: 2915 and 2848 (CH2 stretch), 1465 (CH2 bend), 718 with a 730 shoulder (CH2 rock, crystalline PE). The broad 1031 band is silicate from the soil. Why the library fails: its LDPE reference was measured in reflectance, so its band shapes differ from an ATR measurement. |
| B3 | **Nitrocellulose**. PNNL reference (public domain, distributed with NIST's IARPA set), total reflectance converted to log(1/R). | Best 0.49, second 0.49: **no reliable match** | Not in the library, and the tool says so. Two hits nearly tied is a classic sign that nothing really matches. The nitrate-ester pattern is still visible: 1691, 1301, 867 (transmission literature about 1650, 1280, 840). Reflectance shifts and distorts strong bands. |
| B4 | **Nothing**: a corner pixel of the same map, where there is no particle. | Best 0.27: **no reliable match** | Recognise an empty measurement: noise, water-vapour lines at 3500-3900, CO2 as a **negative** band near 2350 (less CO2 during the sample scan than during the background), and the substrate edge below 800. If you saw a particle there under the microscope, it may be inorganic or metal: send it to SEM-EDX. |
| B5 | Open Specy labels this file **PVA**, but the spectrum does not show PVA's main bands (broad O-H 3200-3400, 1420, 1090). It is consistent with a **clay-type silicate filler** (sharp O-H 3684 with a shoulder near 3620, Si-O 1018, Al-OH 918, kaolinite-like) **plus an acetate-containing polymer** (ester C=O 1736, C-O 1242, as in partly hydrolysed PVA or PVAc). This is my interpretation. | Top hit barrerite, a silicate mineral (0.51); PVA 3rd (0.37): **weak** | A label is not proof, and real samples are mixtures. The library caught the silicate part. Your band reading explains the rest. |

## Interview lines that come out of these exercises

- "The library hit is a hypothesis. On a polyethylene sample the top hit was polystyrene,
  because the reference was measured in reflectance and the sample by ATR. Four bands were
  enough to correct it."
- "When the two best hits are nearly tied, I report *no match* and say what the bands suggest."
- "Before I interpret anything I look for CO2 and water vapour. A negative CO2 band tells me the
  background is stale."
