#############################
# Domain Mapping for Cloud Run
#############################
resource "google_cloud_run_domain_mapping" "domain_mapping" {
  name     = var.domain_name
  location = var.region

  spec {
    route_name       = google_cloud_run_v2_service.service.name
    certificate_mode = "AUTOMATIC"
  }
  metadata {
    namespace = google_cloud_run_v2_service.service.project
  }
}
