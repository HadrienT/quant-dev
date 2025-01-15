

resource "google_monitoring_alert_policy" "idle_alert" {
  display_name = "Idle VM Alert"
  combiner     = "OR"
  enabled      = true
  conditions {
    display_name = "CPU usage < 10%"
    condition_threshold {
      filter          = "metric.type=\"compute.googleapis.com/instance/cpu/utilization\" AND resource.labels.instance_id = \"${google_compute_instance.vm.id}\""
      comparison      = "COMPARISON_LT"
      duration        = "3600s"
      threshold_value = 0.1
      aggregations {
        alignment_period     = "60s"
        per_series_aligner   = "ALIGN_MEAN"
        cross_series_reducer = "REDUCE_MEAN"
      }
    }
  }
  notification_channels = [google_monitoring_notification_channel.function_channel.id]
}

resource "google_monitoring_notification_channel" "function_channel" {
  display_name = "Function Notification Channel"
  type         = "webhook"
  labels = {
    url = google_cloudfunctions_function.auto_shutdown.https_trigger_url
  }
}


resource "google_storage_bucket" "auto_shutdown_function_code_bucket" {
  name          = "auto-shutdown-function-code"
  location      = var.region
  force_destroy = true
}

resource "google_storage_bucket_object" "auto_shutdown_function_code" {
  name   = "function.zip"
  bucket = google_storage_bucket.auto_shutdown_function_code_bucket.name
  source = "${path.module}/scripts/autoshutdown.zip"

}

resource "google_cloudfunctions_function" "auto_shutdown" {
  name                  = "auto-shutdown-vm"
  runtime               = "python311"
  available_memory_mb   = 128
  source_archive_bucket = google_storage_bucket.auto_shutdown_function_code_bucket.name
  source_archive_object = google_storage_bucket_object.auto_shutdown_function_code.name
  entry_point           = "shutdown_vm"
  region                = var.region

  environment_variables = {
    PROJECT_ID = var.project_id
    ZONE       = var.zone
    VM_NAME    = var.vm_name
  }

  trigger_http = true
}

# resource "google_cloud_scheduler_job" "auto_shutdown_job" {
#   name        = "auto-shutdown-scheduler"
#   schedule    = "*/5 * * * *" 
#   description = "Check idle VM and trigger auto shutdown"

#   http_target {
#     uri         = google_cloudfunctions_function.auto_shutdown.https_trigger_url
#     http_method = "POST"
#   }
# }
