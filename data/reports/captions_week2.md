# Caption generation report (Week 2 pilot)

- **Images captioned:** 60 (coco 15, taco 30, zerowaste 15)
- **Models:** blip-base, smolvlm-256m, template
- **Image ids:** `data/captions/pilot_ids.txt`

Auto-check columns come from keyword rules in `scripts/captionlib.py`. They are signals for the human reviewer, not ground truth (see WEEK2_UPDATES.md, section 6).

## Image-level captions

| metric | blip-base | smolvlm-256m | template |
|---|---:|---:|---:|
| captions | 60 | 60 | 60 |
| mean words | 7.0 | 24.6 | 12.8 |
| empty % | 0.0 | 0.0 | 0.0 |
| CLIPScore | 0.747 | 0.745 | 0.71 |
| mentions a material % | 23.3 | 45.0 | 68.3 |
| material recall | 0.181 | 0.355 | 0.821 |
| material conflict % | 1.7 | 1.7 | 0.0 |
| category recall | 0.064 | 0.215 | 1.0 |
| unsupported category % | 6.7 | 3.3 | 0.0 |
| mentions a condition % | 6.7 | 6.7 | 0.0 |
| brand / label text % | 0.0 | 1.7 | 0.0 |
| people / animals mentioned % (TACO+ZeroWaste) | 13.3 | 2.2 | 0.0 |
| sec / caption | 0.35 | 0.66 | 0.0 |
| CLIPScore (coco) | 0.758 | 0.74 | 0.548 |
| CLIPScore (taco) | 0.745 | 0.737 | 0.748 |
| CLIPScore (zerowaste) | 0.74 | 0.764 | 0.794 |

Most frequent unsupported categories: **blip-base** [('bag_wrapper', 1), ('bottle_jar', 1), ('food_waste', 1), ('straw_utensil', 1)]; **smolvlm-256m** [('food_waste', 1), ('straw_utensil', 1)]; **template** []

## Object-level captions (box crops)

| metric | blip-base | smolvlm-256m | template |
|---|---:|---:|---:|
| captions | 175 | 175 | 175 |
| mean words | 6.5 | 14.9 | 9.4 |
| empty % | 0.0 | 0.0 | 0.0 |
| CLIPScore | 0.655 | 0.674 | 0.611 |
| material match % | 19.7 | 95.8 | 100.0 |
| material conflict % | 2.3 | 0.0 | 0.0 |
| category match % | 13.1 | 98.5 | 100.0 |
| mentions a condition % | 0.6 | 17.7 | 0.0 |
| JSON format followed % | – | – | – |
| brand / label text % | 0.0 | 3.4 | 0.0 |
| sec / caption | 0.29 | 0.58 | 0.0 |
| CLIPScore (coco) | 0.66 | 0.662 | 0.563 |
| CLIPScore (taco) | 0.696 | 0.708 | 0.644 |
| CLIPScore (zerowaste) | 0.625 | 0.654 | 0.603 |

Conditions proposed by the VLMs (object level): **blip-base** {'broken': 1}; **smolvlm-256m** {'intact': 23, 'torn': 5, 'soiled': 2, 'crumpled': 1}

## Head-to-head CLIPScore (blip-base vs smolvlm-256m)

| level | blip-base | smolvlm-256m | tie |
|---|---:|---:|---:|
| image | 29 | 31 | 0 |
| object | 75 | 100 | 0 |

## Examples

Rendered in `data/reports/caption_examples.jpg`.

**coco_000000304984** (coco; labels: {'cup': 1})

- *blip-base* (CLIPScore 0.6393): a plate of food with a fork and knife
- *smolvlm-256m* (CLIPScore 0.664): The image shows a plate of food, specifically a sandwich, placed on a table. The sandwich is made with ham, cheese, and lettuce, and is served with a side of coleslaw.
- *template* (CLIPScore 0.4571): One item: a wine glass.
- largest object `coco_000000304984_o000` (wine glass):
  - *blip-base*: a plate of food
  - *smolvlm-256m*: The wine glass is a clear glass with a rim that is slightly curved.
  - *template*: wine glass in the top right of the image

**coco_000000116206** (coco; labels: {'cup': 1})

- *blip-base* (CLIPScore 0.8903): chopped vegetables on a cutting board
- *smolvlm-256m* (CLIPScore 0.7535): The image shows a wooden cutting board placed on a black surface, likely a table or countertop. On the cutting board, there are various ingredients and ingredients are mixed together.
- *template* (CLIPScore 0.5655): One item: a cup.
- largest object `coco_000000116206_o000` (cup):
  - *blip-base*: a bowl of vegetables
  - *smolvlm-256m*: The cup is a clear glass with red markings.
  - *template*: cup in the top center of the image

**taco_00048** (taco; labels: {'carton_box': 1})

- *blip-base* (CLIPScore 0.6712): a piece of paper on the floor
- *smolvlm-256m* (CLIPScore 0.6647): This image shows a close-up view of a section of a wall, specifically a section of a wall that is made of large, irregularly shaped tiles. The tiles are a mix of brown and beige, with some areas having a darker hue and others lighter.
- *template* (CLIPScore 0.6561): One waste item indoors: an egg carton.
- largest object `taco_00048_o000` (Egg carton):
  - *blip-base*: a piece of paper with a piece missing
  - *smolvlm-256m*: The egg carton is brown and appears to be made of cardboard.
  - *template*: egg carton in the middle left of the image

**taco_01194** (taco; labels: {'carton_box': 2, 'bag_wrapper': 1, 'cigarette': 1})

- *blip-base* (CLIPScore 0.7447): a field with a lot of trash
- *smolvlm-256m* (CLIPScore 0.7966): The image shows a patch of green grass and weeds in the foreground, with a few scattered rocks and small plants. The grass and weeds are interspersed with patches of dirt and soil, indicating a natural environment.
- *template* (CLIPScore 0.7672): Four waste items among vegetation: two other cartons, a plastic film and a cigarette.
- largest object `taco_01194_o002` (Plastic film):
  - *blip-base*: a spering weed in the grass
  - *smolvlm-256m*: The plastic film is bright green in colour and appears to be a natural material.
  - *template*: plastic film in the top right of the image

**zerowaste_val_00126** (zerowaste; labels: {'unspecified': 3, 'carton_box': 3})

- *blip-base* (CLIPScore 0.7783): a pile of paper and other items
- *smolvlm-256m* (CLIPScore 0.79): The image shows a pile of garbage, including plastic bags, food wrappers, and other items, scattered across a dark, possibly wet surface. The garbage is primarily made of plastic bags, which are commonly used for transporting food and other items.
- *template* (CLIPScore 0.7538): Six waste items on a recycling conveyor belt: three pieces of soft plastic film and three pieces of cardboard.
- largest object `zerowaste_val_00126_o002` (soft_plastic):
  - *blip-base*: a garbage bag and a bag of chips
  - *smolvlm-256m*: The soft plastic film is transparent and contains a small amount of liquid.
  - *template*: piece of soft plastic film in the middle right of the image

**zerowaste_val_00119** (zerowaste; labels: {'carton_box': 3, 'unspecified': 1})

- *blip-base* (CLIPScore 0.658): the trash bins in the parking
- *smolvlm-256m* (CLIPScore 0.8662): The image shows a cluttered and somewhat disorganized floor. The floor is covered with a variety of items, including plastic covers, plastic bags, and a few objects that appear to be food items.
- *template* (CLIPScore 0.7848): Four waste items on a recycling conveyor belt: three pieces of cardboard and a piece of soft plastic film.
- largest object `zerowaste_val_00119_o001` (soft_plastic):
  - *blip-base*: a trash bin
  - *smolvlm-256m*: It is a plastic film that is in the shape of a bubble.
  - *template*: piece of soft plastic film in the middle left of the image
