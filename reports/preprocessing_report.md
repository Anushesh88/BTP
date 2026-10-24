# Phase 4: Preprocessing & Sliding Window Generation Report

> **Objective**: Preserve native resolution, apply edge-preserving denoising, color normalization, CLAHE contrast enhancement, unsharp masking, and generate overlapping sliding windows with exact coordinate tracking.

## Configuration Parameters
- **Window Size**: 512 x 512 pixels
- **Stride**: 256 pixels (Overlap: 50.0%)
- **Denoising Filter**: median
- **Lighting & Contrast**: CLAHE (clip=2.0, grid=8x8) + Gamma (1.05)
- **Sharpening**: Unsharp Mask (strength=1.2)
- **Border Handling**: Reflective padding for marginalia and borders

## Processed Paintings Summary

| Painting ID | Original Canvas | Extracted Windows | Enhanced Image Path |
| --- | --- | --- | --- |
| a_woman_with_two_children_having_abandoned_her_home_goes_into_the_forest_where_she_encounters_a_leopard_from_a_tuti-nama_c_1560 | 2250x3400 | 104 | `data\processed\preprocessed_images\a_woman_with_two_children_having_abandoned_her_home_goes_into_the_forest_where_she_encounters_a_leopard_from_a_tuti-nama_c_1560_enhanced.jpg` |
| bashir_confides_his_love_for_habbaza_to_an_arab_friend_and_sends_him_to_her_with_a_message_from_a_tuti-nama_tales_of_a_p_c_1560 | 2209x3400 | 104 | `data\processed\preprocessed_images\bashir_confides_his_love_for_habbaza_to_an_arab_friend_and_sends_him_to_her_with_a_message_from_a_tuti-nama_tales_of_a_p_c_1560_enhanced.jpg` |
| habbaza_meets_bashir_under_a_tree_from_a_tuti-nama_tales_of_a_parrot_twenty-fourth_night_c_1560 | 2046x2702 | 70 | `data\processed\preprocessed_images\habbaza_meets_bashir_under_a_tree_from_a_tuti-nama_tales_of_a_parrot_twenty-fourth_night_c_1560_enhanced.jpg` |
| habbazas_sister_who_is_sent_to_console_her_discovers_the_disguised_arab_in_her_place_from_a_tuti-nama_tales_of_a_parrot__c_1560 | 2057x2263 | 64 | `data\processed\preprocessed_images\habbazas_sister_who_is_sent_to_console_her_discovers_the_disguised_arab_in_her_place_from_a_tuti-nama_tales_of_a_parrot__c_1560_enhanced.jpg` |
| kaiwan_sends_a_message_of_love_to_khurshid_wife_of_his_brother_utarid_who_is_away_on_a_journey_from_a_tuti-nama_tales_of_c_1560 | 2099x1763 | 48 | `data\processed\preprocessed_images\kaiwan_sends_a_message_of_love_to_khurshid_wife_of_his_brother_utarid_who_is_away_on_a_journey_from_a_tuti-nama_tales_of_c_1560_enhanced.jpg` |

- **Total Sliding Windows Generated**: 390
- **Windows Manifest Location**: [`data\processed\windows_manifest.jsonl`](data\processed\windows_manifest.jsonl)

## Inspection Graphic
- See visual inspection comparison: [`preprocessing_inspection_a_woman_with_two_children_having_abandoned_her_home_goes_into_the_forest_where_she_encounters_a_leopard_from_a_tuti-nama_c_1560.png`](preprocessing_inspection_a_woman_with_two_children_having_abandoned_her_home_goes_into_the_forest_where_she_encounters_a_leopard_from_a_tuti-nama_c_1560.png)
