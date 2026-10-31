# CaptionGene dataset statistics (Week 1, pre-captioning)

- **Images:** 2410  |  **Objects:** 9391  |  **Mean objects/image:** 3.9  |  **Max objects/image:** 90

## Images per source and split

| source | train | val | test | total |
|---|---:|---:|---:|---:|
| taco | 1063 | 222 | 208 | 1493 |
| zerowaste | 418 | 90 | 89 | 597 |
| coco | 231 | 43 | 46 | 320 |

## Objects per waste category

| category | train | val | test | total |
|---|---:|---:|---:|---:|
| carton_box | 1871 | 387 | 369 | 2627 |
| unspecified | 843 | 219 | 172 | 1234 |
| other_litter | 821 | 156 | 150 | 1127 |
| bag_wrapper | 585 | 163 | 122 | 870 |
| bottle_jar | 553 | 145 | 130 | 828 |
| cup | 582 | 117 | 113 | 812 |
| cigarette | 420 | 107 | 133 | 660 |
| cap_lid | 263 | 61 | 51 | 375 |
| can | 189 | 37 | 47 | 273 |
| straw_utensil | 150 | 28 | 20 | 198 |
| paper | 108 | 21 | 24 | 153 |
| glass_fragment | 108 | 3 | 27 | 138 |
| container_packaging | 60 | 13 | 13 | 86 |
| food_waste | 6 | 1 | 1 | 8 |
| hazardous | 2 | 0 | 0 | 2 |

## Objects per material

| material | train | val | test | total |
|---|---:|---:|---:|---:|
| plastic | 2350 | 580 | 454 | 3384 |
| paper_cardboard | 1966 | 410 | 391 | 2767 |
| unknown | 950 | 211 | 185 | 1346 |
| composite | 532 | 126 | 166 | 824 |
| metal | 409 | 84 | 99 | 592 |
| glass | 321 | 39 | 72 | 432 |
| textile_rubber | 25 | 7 | 4 | 36 |
| organic | 6 | 1 | 1 | 8 |
| e_waste | 2 | 0 | 0 | 2 |

## Objects per default sorting action

| sorting_action | train | val | test | total |
|---|---:|---:|---:|---:|
| landfill | 2277 | 536 | 497 | 3310 |
| recycle_paper | 1966 | 404 | 395 | 2765 |
| needs_review | 1088 | 226 | 210 | 1524 |
| recycle_plastic | 713 | 176 | 145 | 1034 |
| recycle_metal | 409 | 84 | 99 | 592 |
| recycle_glass | 70 | 20 | 20 | 110 |
| compost | 36 | 12 | 6 | 54 |
| hazardous_dropoff | 2 | 0 | 0 | 2 |

## Images per clutter level

| clutter | train | val | test | total |
|---|---:|---:|---:|---:|
| single | 560 | 113 | 100 | 773 |
| sparse | 507 | 97 | 117 | 721 |
| moderate | 522 | 114 | 106 | 742 |
| dense | 123 | 31 | 20 | 174 |

## Images per scene environment (an image may have several)

| environment | train | val | test | total |
|---|---:|---:|---:|---:|
| vegetation | 536 | 107 | 111 | 754 |
| conveyor_belt | 418 | 90 | 89 | 597 |
| sand_dirt_pebbles | 436 | 82 | 79 | 597 |
| pavement_urban | 296 | 75 | 62 | 433 |
| unknown | 302 | 55 | 58 | 415 |
| indoor_manmade | 88 | 13 | 11 | 112 |
| water | 63 | 11 | 10 | 84 |
| clean_background | 25 | 7 | 5 | 37 |
| trash_pile | 4 | 0 | 1 | 5 |

## Object size and flags

| size | train | val | test | total |
|---|---:|---:|---:|---:|
| small | 2883 | 624 | 587 | 4094 |
| medium | 3044 | 691 | 636 | 4371 |
| large | 634 | 143 | 149 | 926 |

Object flags: {'possibly_occluded': 2049, 'tiny': 1205}  

Image quality flags: {'too_dark': 9}
