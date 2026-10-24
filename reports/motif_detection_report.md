# Phase 5: Open-Vocabulary Motif & Region Detection Report

> **Objective**: Run zero-shot open-vocabulary detection across overlapping sliding windows, merge proposals with global Non-Maximum Suppression (NMS), classify into coarse taxonomies, crop motif patches, and evaluate fine-tuning necessity.

## Detection Pipeline Configuration
- **Detector Architecture**: Open-Vocabulary YOLO-World (yolov8s-worldv2.pt)
- **Vocabulary Size**: 30 plain-English prompts
- **Canvas**: longest side capped at 1280px; 640px windows with 50% overlap
- **Paintings Processed**: 564; **Motif Crops**: 9609
- **Confidence Threshold**: 0.15
- **Global NMS IoU Threshold**: 0.40

## Category Distribution of Extracted Motifs

- **Instrument_object**: 2796 crops extracted
- **Architecture**: 2647 crops extracted
- **Animal**: 1740 crops extracted
- **Human**: 1725 crops extracted
- **Flora**: 700 crops extracted
- **Text**: 1 crops extracted

## Summary by Painting (first 25 of 564)

| Painting ID | Raga | Motifs Count | Detected Labels |
| --- | --- | --- | --- |
| a_woman_with_two_children_having_abandoned_her_home_goes_into_the_forest_where_she_encounters_a_leopard_from_a_tuti-nama_c_1560 | Non-Ragamala | 13 | carpet, bird, bird, peacock... |
| bashir_confides_his_love_for_habbaza_to_an_arab_friend_and_sends_him_to_her_with_a_message_from_a_tuti-nama_tales_of_a_p_c_1560 | Non-Ragamala | 32 | turban, carpet, building, turban... |
| habbaza_meets_bashir_under_a_tree_from_a_tuti-nama_tales_of_a_parrot_twenty-fourth_night_c_1560 | Non-Ragamala | 45 | turban, turban, turban, turban... |
| habbazas_sister_who_is_sent_to_console_her_discovers_the_disguised_arab_in_her_place_from_a_tuti-nama_tales_of_a_parrot__c_1560 | Non-Ragamala | 41 | turban, pavilion, pavilion, turban... |
| kaiwan_sends_a_message_of_love_to_khurshid_wife_of_his_brother_utarid_who_is_away_on_a_journey_from_a_tuti-nama_tales_of_c_1560 | Non-Ragamala | 30 | building, pavilion, pavilion, pavilion... |
| kamjuy_the_wife_of_the_raja_averts_her_face_from_the_fishes_from_a_tuti-nama_tales_of_a_parrot_twenty-third_night_c_1560 | Non-Ragamala | 19 | pavilion, turban, building, building... |
| khujasta_kills_the_pet_mynah_who_advises_her_not_to_be_unfaithful_to_maymun_her_husband_from_a_tuti-nama_tales_of_the_pa_c_1560 | Non-Ragamala | 15 | turban, building, building, building... |
| khulasa_a_vizier_sees_the_daughter_of_khassa_another_vizier_and_covets_her_from_a_tuti-nama_tales_of_a_parrot_fifty-firs_c_1560 | Non-Ragamala | 18 | peacock, turban, building, tiger... |
| khurshid_reunited_with_her_husband_utarid_from_a_tuti-nama_tales_of_a_parrot_thirty-second_night_c_1560 | Non-Ragamala | 37 | pavilion, peacock, pavilion, building... |
| khusrau_the_king_of_kings_pays_homage_to_the_pious_daughter_of_khassa_from_a_tuti-nama_tales_of_a_parrot_fifty-first_nig_c_1560 | Non-Ragamala | 33 | turban, turban, horse, peacock... |
| latif_who_has_murdered_his_brother_falsely_accuses_khurshid_of_the_deed_from_a_tuti-nama_tales_of_a_parrot_thirty-second_c_1560 | Non-Ragamala | 28 | turban, building, building, building... |
| mahrusa_kills_herself_at_the_tomb_of_the_king_of_zabul_and_her_husband_does_likewise_from_a_tuti-nama_tales_of_a_parrot__c_1560 | Non-Ragamala | 32 | turban, pavilion, tree, pavilion... |
| mahrusas_marriage_to_the_prefect_of_the_city_from_a_tuti-nama_tales_of_a_parrot_thirty-sixth_night_c_1560 | Non-Ragamala | 48 | turban, turban, turban, turban... |
| mukhtar_throws_his_wife_maimuna_into_the_pit_but_she_saves_herself_from_a_tuti-nama_tales_of_a_parrot_twenty-fifth_night_c_1560 | Non-Ragamala | 13 | turban, turban, turban, peacock... |
| nikfal_the_fortune_of_the_prince_in_the_form_of_a_woman_offers_to_accompany_him_from_a_tuti-nama_tales_of_a_parrot_eight_c_1560 | Non-Ragamala | 23 | peacock, turban, woman, peacock... |
| preparation_for_the_marriage_of_mahmuda_to_the_young_vizier_from_a_tuti-nama_tales_of_a_parrot_thirty-third_night_c_1560 | Non-Ragamala | 24 | turban, peacock, pavilion, turban... |
| repenting_his_conduct_ubaid_falls_at_the_feet_of_his_parents_from_a_tuti-nama_tales_of_a_parrot_forty-second_night_c_1560 | Non-Ragamala | 25 | woman, building, peacock, turban... |
| shahr-arai_and_her_husband_adopt_her_lover_as_a_brother_in_the_family_from_a_tuti-nama_tales_of_a_parrot_fortieth_night_c_1560 | Non-Ragamala | 22 | turban, turban, building, pavilion... |
| shahr-arai_and_her_lover_dallying_on_a_bed_beneath_which_is_concealed_her_husband_from_a_tuti-nama_tales_of_a_parrot_for_c_1560 | Non-Ragamala | 25 | building, building, building, turban... |
| shahr-arais_husband_bends_to_kiss_his_wife_who_feigns_sleep_from_a_tuti-nama_tales_of_a_parrot_fortieth_night_c_1560 | Non-Ragamala | 19 | building, building, building, carpet... |
| the_bird_of_seven_colors_brings_a_sable_to_the_pious_man_from_a_tuti-nama_tales_of_a_parrot_fifty-second_night_c_1560 | Non-Ragamala | 9 | turban, peacock, peacock, turban... |
| the_brahman_comes_upon_a_lion_who_has_a_deer_and_a_gazelle_as_his_viziers_from_a_tuti-nama_tales_of_a_parrot_twenty-firs_c_1560 | Non-Ragamala | 26 | turban, peacock, cow, peacock... |
| the_brahman_unable_to_select_from_the_four_gifts_of_the_king_of_the_ocean_seeks_the_rajas_advice_from_a_tuti-nama_tales__c_1560 | Non-Ragamala | 28 | horse, peacock, turban, pavilion... |
| the_brahmans_wife_who_killed_a_peacock_and_ate_its_gallbladder_on_the_physicians_advice_from_a_tuti-nama_tales_of_a_parr_c_1560 | Non-Ragamala | 20 | turban, peacock, musician, man... |
| the_cat_attacks_the_mice_which_disturb_the_lion_from_a_tuti-nama_tales_of_a_parrot_fifteenth_night_c_1560 | Non-Ragamala | 18 | bird, tree, bird, tree... |

## Caption-Grounded Recall & Fine-Tuning Recommendation
> The dataset provides no bounding boxes, so recall is an image-level proxy: of the prompt objects named in a painting's caption, how many were detected anywhere in it.
- **Caption-grounded recall**: 47.4% (566/1194 mentions)
- **Decision**: **Caption-grounded recall 47.4% is below 60%; fine-tune on box labels once annotated.**

| Prompt | Recall | Captions mentioning |
| --- | --- | --- |
| clouds | 0% | 79 |
| text | 0% | 54 |
| lotus | 0% | 14 |
| crown | 12% | 33 |
| musical instrument | 12% | 8 |
| lamp | 17% | 18 |
| deer | 19% | 16 |
| snake | 20% | 5 |
| man | 24% | 59 |
| sword | 26% | 68 |
| flower | 30% | 83 |
| woman | 35% | 40 |
| carpet | 40% | 50 |
| tiger | 40% | 5 |
| child | 40% | 5 |
| canopy | 42% | 36 |
| musician | 43% | 14 |
| elephant | 46% | 13 |
| tree | 48% | 192 |
| water | 50% | 16 |
| bird | 58% | 64 |
| horse | 62% | 39 |
| peacock | 88% | 17 |
| pavilion | 95% | 42 |
| turban | 96% | 197 |
| building | 100% | 25 |

## Inspection Graphic
- Visual bounding box overlay: [`motif_detection_a_woman_with_two_children_having_abandoned_her_home_goes_into_the_forest_where_she_encounters_a_leopard_from_a_tuti-nama_c_1560.png`](motif_detection_a_woman_with_two_children_having_abandoned_her_home_goes_into_the_forest_where_she_encounters_a_leopard_from_a_tuti-nama_c_1560.png)
- Master Motifs Manifest: [`data\processed\motifs\detected_motifs_manifest.jsonl`](data\processed\motifs\detected_motifs_manifest.jsonl)
