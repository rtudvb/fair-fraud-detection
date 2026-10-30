| depth | rule | surrogate_flags | coverage | model_flag_rate | fraud_rate | fraud_share |
| --- | --- | --- | --- | --- | --- | --- |
| 3 | housing_status = BA 且 has_other_cards ≤ 0.50 且 device_os = windows | True | 0.0476 | 0.4808 | 0.0997 | 0.3384 |
| 3 | housing_status = BA 且 has_other_cards ≤ 0.50 且 device_os ≠ windows | True | 0.0993 | 0.1599 | 0.0336 | 0.2380 |
| 3 | housing_status = BA 且 has_other_cards > 0.50 且 device_os = windows | True | 0.0125 | 0.1252 | 0.0257 | 0.0229 |
| 3 | housing_status ≠ BA 且 device_os = windows 且 current_address_months_count > 33.50 | True | 0.1088 | 0.0896 | 0.0197 | 0.1525 |
| 3 | housing_status ≠ BA 且 device_os ≠ windows 且 device_distinct_emails_8w > 1.50 | False | 0.0074 | 0.0654 | 0.0159 | 0.0083 |
| 3 | housing_status = BA 且 has_other_cards > 0.50 且 device_os ≠ windows | False | 0.0351 | 0.0287 | 0.0085 | 0.0212 |
| 3 | housing_status ≠ BA 且 device_os = windows 且 current_address_months_count ≤ 33.50 | False | 0.0778 | 0.0186 | 0.0077 | 0.0427 |
| 3 | housing_status ≠ BA 且 device_os ≠ windows 且 device_distinct_emails_8w ≤ 1.50 | False | 0.6114 | 0.0061 | 0.0040 | 0.1758 |
| 4 | housing_status = BA 且 has_other_cards ≤ 0.50 且 device_os = windows 且 prev_address_months_count ≤ 3.00 | True | 0.0409 | 0.5310 | 0.1083 | 0.3158 |
| 4 | housing_status = BA 且 has_other_cards ≤ 0.50 且 device_os ≠ windows 且 keep_alive_session ≤ 0.50 | True | 0.0418 | 0.2769 | 0.0505 | 0.1505 |
| 4 | housing_status = BA 且 has_other_cards ≤ 0.50 且 device_os = windows 且 prev_address_months_count > 3.00 | True | 0.0067 | 0.1739 | 0.0473 | 0.0226 |
| 4 | housing_status ≠ BA 且 device_os = windows 且 current_address_months_count ≤ 33.50 且 device_distinct_emails_8w > 1.50 | True | 0.0011 | 0.1918 | 0.0457 | 0.0035 |
| 4 | housing_status = BA 且 has_other_cards > 0.50 且 device_os = windows 且 keep_alive_session ≤ 0.50 | True | 0.0069 | 0.1920 | 0.0398 | 0.0195 |
| 4 | housing_status ≠ BA 且 device_os = windows 且 current_address_months_count > 33.50 且 keep_alive_session ≤ 0.50 | True | 0.0436 | 0.1524 | 0.0303 | 0.0942 |
| 4 | housing_status = BA 且 has_other_cards ≤ 0.50 且 device_os ≠ windows 且 keep_alive_session > 0.50 | True | 0.0576 | 0.0750 | 0.0214 | 0.0876 |
| 4 | housing_status ≠ BA 且 device_os ≠ windows 且 device_distinct_emails_8w > 1.50 且 keep_alive_session ≤ 0.50 | True | 0.0049 | 0.0894 | 0.0181 | 0.0063 |
| 4 | housing_status = BA 且 has_other_cards > 0.50 且 device_os ≠ windows 且 keep_alive_session ≤ 0.50 | False | 0.0180 | 0.0479 | 0.0127 | 0.0163 |
| 4 | housing_status ≠ BA 且 device_os = windows 且 current_address_months_count > 33.50 且 keep_alive_session > 0.50 | False | 0.0652 | 0.0476 | 0.0126 | 0.0584 |
| 4 | housing_status ≠ BA 且 device_os ≠ windows 且 device_distinct_emails_8w > 1.50 且 keep_alive_session > 0.50 | False | 0.0025 | 0.0193 | 0.0116 | 0.0021 |
| 4 | housing_status ≠ BA 且 device_os ≠ windows 且 device_distinct_emails_8w ≤ 1.50 且 device_os = macintosh | False | 0.0413 | 0.0300 | 0.0096 | 0.0281 |
| 4 | housing_status = BA 且 has_other_cards > 0.50 且 device_os = windows 且 keep_alive_session > 0.50 | False | 0.0056 | 0.0440 | 0.0086 | 0.0035 |
| 4 | housing_status ≠ BA 且 device_os = windows 且 current_address_months_count ≤ 33.50 且 device_distinct_emails_8w ≤ 1.50 | False | 0.0767 | 0.0162 | 0.0072 | 0.0393 |
| 4 | housing_status = BA 且 has_other_cards > 0.50 且 device_os ≠ windows 且 keep_alive_session > 0.50 | False | 0.0171 | 0.0085 | 0.0040 | 0.0049 |
| 4 | housing_status ≠ BA 且 device_os ≠ windows 且 device_distinct_emails_8w ≤ 1.50 且 device_os ≠ macintosh | False | 0.5701 | 0.0043 | 0.0036 | 0.1477 |
