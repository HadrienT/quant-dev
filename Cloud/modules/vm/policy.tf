
#############################
# Alert policy for idle VM
#############################
# resource "google_monitoring_alert_policy" "idle_alert" {
#   display_name = "Idle VM Alert"
#   combiner     = "OR"
#   enabled      = true

#   conditions {
#     display_name = "CPU usage < 10%"
#     condition_threshold {
#       filter          = "metric.type=\"compute.googleapis.com/instance/cpu/utilization\" AND resource.labels.instance_id = \"${google_compute_instance.instance.id}\""
#       comparison      = "COMPARISON_LT"
#       duration        = "3600s"
#       threshold_value = 0.1
#       aggregations {
#         alignment_period     = "60s"
#         per_series_aligner   = "ALIGN_MEAN"
#         cross_series_reducer = "REDUCE_MEAN"
#       }
#     }
#   }
#   notification_channels = [google_monitoring_notification_channel.function_channel.id]
# }

# resource "google_monitoring_notification_channel" "function_channel" {
#   display_name = "Function Notification Channel"
#   type         = "webhook"
#   labels = {
#     url = google_cloudfunctions_function.auto_shutdown.https_trigger_url
#   }
# }
