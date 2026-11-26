# Caption quality report: zerowaste

- **Captioned images:** 597
- **Caption sources:** zw-detailed, template, smolvlm-256m [chat_image_zw_short], smolvlm-256m, blip-base
- **Quality score weights:** count 25, material 20, no_hallucination 20, detail 15, clip 10, fluency 10

## What each parameter measures

| parameter | meaning | good value |
|---|---|---|
| count stated % | caption gives how many waste items there are | high |
| count correct % | stated count equals the number of annotated objects | high |
| count error | mean difference between stated and annotated count | 0 |
| material count acc. | share of materials (cardboard, plastic, metal) whose count is right | 1 |
| material recall | share of annotated materials named in the caption | 1 |
| hallucination % | mentions people/animals, non-waste objects (keyboard, book), a wrong setting (table, floor, street) or a category/material that is not annotated | 0 |
| repetitive % | caption contains a repetition loop | 0 |
| detail score | share of {count, material, location, colour, condition, setting} mentioned | 1 |
| CLIPScore | image-text similarity from CLIP ViT-B/32 (text is cut at 77 tokens) | 0.8+ |
| quality (0-100) | weighted mean of count, material, no-hallucination, detail, CLIP, fluency | high |

## All captioned images

| model | captions | quality (0-100) | count stated % | count correct % | count error | material count acc. | material recall | hallucination % | repetitive % | detail score | CLIPScore | words |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| zw-detailed | 597 | 93.2 | 100.0 | 100.0 | 0 | 1.0 | 1.0 | 0.0 | 0.0 | 0.83 | 0.766 | 82.0 |
| template | 15 | 89.8 | 100.0 | 100.0 | 0 | 1.0 | 1.0 | 0.0 | 0.0 | 0.5 | 0.794 | 18.4 |
| smolvlm-256m [chat_image_zw_short] | 15 | 56.6 | 66.7 | 26.7 | 2.8 | 0.4 | 0.67 | 0.0 | 6.7 | 0.22 | 0.608 | 8.3 |
| smolvlm-256m | 15 | 41.7 | 13.3 | 0.0 | 5.5 | 0.0 | 0.62 | 53.3 | 20.0 | 0.36 | 0.764 | 30.2 |
| blip-base | 15 | 37.3 | 6.7 | 0.0 | 2 | 0.0 | 0.32 | 33.3 | 0.0 | 0.1 | 0.74 | 7.3 |

## Same 15 images for every caption source (fair comparison)

| model | captions | quality (0-100) | count stated % | count correct % | count error | material count acc. | material recall | hallucination % | repetitive % | detail score | CLIPScore | words |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| zw-detailed | 15 | 93.3 | 100.0 | 100.0 | 0 | 1.0 | 1.0 | 0.0 | 0.0 | 0.83 | 0.765 | 84.3 |
| template | 15 | 89.8 | 100.0 | 100.0 | 0 | 1.0 | 1.0 | 0.0 | 0.0 | 0.5 | 0.794 | 18.4 |
| smolvlm-256m [chat_image_zw_short] | 15 | 56.6 | 66.7 | 26.7 | 2.8 | 0.4 | 0.67 | 0.0 | 6.7 | 0.22 | 0.608 | 8.3 |
| smolvlm-256m | 15 | 41.7 | 13.3 | 0.0 | 5.5 | 0.0 | 0.62 | 53.3 | 20.0 | 0.36 | 0.764 | 30.2 |
| blip-base | 15 | 37.3 | 6.7 | 0.0 | 2 | 0.0 | 0.32 | 33.3 | 0.0 | 0.1 | 0.74 | 7.3 |

## Examples (best two and worst caption per source)

**zw-detailed**

- `zerowaste_train_02930` (3 objects) **96.7**: A recycling conveyor belt with 3 waste items: 3 pieces of cardboard. The pieces of cardboard are in the centre, at the top and on the left. The largest item is a medium orange piece of cardboard in the centre. The items cover about 5% of the belt. For sorting, 3 go to paper recycling.
- `zerowaste_train_02831` (4 objects) **96.7**: A recycling conveyor belt with 4 waste items: 2 pieces of cardboard and 2 pieces of soft plastic film. The pieces of cardboard are at the top and in the centre and the pieces of soft plastic film are on the right and in the top right. The largest item is a large white piece of cardboard at the top. 
- `zerowaste_val_00063` (2 objects) **87.9**: A recycling conveyor belt with 2 waste items: a piece of soft plastic film and a piece of cardboard. The piece of soft plastic film is in the bottom right and the piece of cardboard is in the bottom left. The largest item is a medium grey piece of soft plastic film in the bottom right. The items cov

**template**

- `zerowaste_val_00560` (9 objects) **92.4**: Nine waste items on a recycling conveyor belt: five pieces of cardboard, three pieces of soft plastic film and a rigid plastic item.
- `zerowaste_val_00470` (6 objects) **91.1**: Six waste items on a recycling conveyor belt: five pieces of cardboard and a rigid plastic item.
- `zerowaste_val_00000` (1 objects) **87.4**: One waste item on a recycling conveyor belt: a piece of cardboard.

**smolvlm-256m [chat_image_zw_short]**

- `zerowaste_val_00332` (6 objects) **84.2**: 4 cardboard pieces, 2 soft plastic film pieces.
- `zerowaste_val_00219` (11 objects) **84.2**: 6 cardboard, 4 plastic, and 1 rigid plastic.
- `zerowaste_val_00169` (9 objects) **20.4**: 7, 2, 2, 7, 2, 2, 7, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,

**smolvlm-256m**

- `zerowaste_val_00470` (6 objects) **64.7**: The image shows a cluttered area with various items scattered on a flat surface. The objects include: 1. **Plastic Wrapper**: There are several pieces of plastic wrapped in white paper, indicating that the area might be a storage or storage area.
- `zerowaste_val_00560` (9 objects) **62.5**: The image shows a pile of paper and plastic waste, which appears to be from a landfill. The waste includes various types of paper, including sheets of paper, magazines, and newspapers.
- `zerowaste_val_00169` (9 objects) **14.9**: A cluttered table with a keyboard and garbage can.

**blip-base**

- `zerowaste_val_00069` (3 objects) **61.2**: a pile of paper and a piece of paper
- `zerowaste_val_00338` (5 objects) **60.0**: a pile of paper and other items
- `zerowaste_val_00470` (6 objects) **12.7**: the aftermath of a house in the bronx neighborhood

## Caveats

- Count, material and hallucination checks compare the caption with the annotation using keyword rules. They are signals for the reviewer, not ground truth.
- `zw-detailed` and `template` are built from the annotation, so their counts are correct by construction. Their score shows how complete and readable they are, not whether a model can count. Their colour words come from the median pixel colour inside each object's mask.
- ZeroWaste only annotates cardboard, soft plastic, rigid plastic and metal. White paper sheets and other unlabelled items on the belt are not counted, so a VLM that counts them is marked wrong.
- Reference-based metrics (BLEU, CIDEr, METEOR) need human-approved captions; they are planned for Week 4 once reviewers have approved captions in the app.
