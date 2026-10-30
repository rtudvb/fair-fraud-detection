| depth | rule | surrogate_flags | coverage | model_flag_rate | fraud_rate | fraud_share |
| --- | --- | --- | --- | --- | --- | --- |
| 4 | housing_status = BA 且 has_other_cards ≤ 0.50 且 device_os = windows 且 prev_address_months_count ≤ 3.00 | True | 0.0409 | 0.5310 | 0.1083 | 0.3158 |
| 3 | housing_status = BA 且 has_other_cards ≤ 0.50 且 device_os = windows | True | 0.0476 | 0.4808 | 0.0997 | 0.3384 |
| 4 | housing_status = BA 且 has_other_cards ≤ 0.50 且 device_os ≠ windows 且 keep_alive_session ≤ 0.50 | True | 0.0418 | 0.2769 | 0.0505 | 0.1505 |
| 4 | housing_status = BA 且 has_other_cards ≤ 0.50 且 device_os = windows 且 prev_address_months_count > 3.00 | True | 0.0067 | 0.1739 | 0.0473 | 0.0226 |
| 4 | housing_status ≠ BA 且 device_os = windows 且 current_address_months_count ≤ 33.50 且 device_distinct_emails_8w > 1.50 | True | 0.0011 | 0.1918 | 0.0457 | 0.0035 |
| 4 | housing_status = BA 且 has_other_cards > 0.50 且 device_os = windows 且 keep_alive_session ≤ 0.50 | True | 0.0069 | 0.1920 | 0.0398 | 0.0195 |
| 3 | housing_status = BA 且 has_other_cards ≤ 0.50 且 device_os ≠ windows | True | 0.0993 | 0.1599 | 0.0336 | 0.2380 |
| 4 | housing_status ≠ BA 且 device_os = windows 且 current_address_months_count > 33.50 且 keep_alive_session ≤ 0.50 | True | 0.0436 | 0.1524 | 0.0303 | 0.0942 |
| 3 | housing_status = BA 且 has_other_cards > 0.50 且 device_os = windows | True | 0.0125 | 0.1252 | 0.0257 | 0.0229 |
| 4 | housing_status = BA 且 has_other_cards ≤ 0.50 且 device_os ≠ windows 且 keep_alive_session > 0.50 | True | 0.0576 | 0.0750 | 0.0214 | 0.0876 |
