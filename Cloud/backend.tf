

resource "google_compute_backend_service" "cloud_run_backend" {
  name                  = "cloud-run-backend"
  load_balancing_scheme = "EXTERNAL"
  protocol              = "HTTPS"
  timeout_sec           = 30
  enable_cdn            = false
  backend {
    group = google_compute_region_network_endpoint_group.cloud_run_neg.id
  }

  # security_policy = google_compute_security_policy.strict_policy.id

  port_name = "http" # Mandatory for Cloud Run

  # Header to check the host
  custom_request_headers = [
    "Host: ${google_cloud_run_v2_service.service.name}-${google_cloud_run_v2_service.service.project}.a.run.app"
  ]

  # Headers to handle CORS
  custom_response_headers = [
    "Access-Control-Allow-Origin: *",
    "Access-Control-Allow-Methods: GET, POST, OPTIONS",
    "Access-Control-Allow-Headers: DNT,User-Agent,X-Requested-With,If-Modified-Since,Cache-Control,Content-Type,Range",
    "Access-Control-Expose-Headers: Content-Length,Content-Range"
  ]

  # Advanced logging to track requests
  log_config {
    enable      = true
    sample_rate = 1.0
  }
}
