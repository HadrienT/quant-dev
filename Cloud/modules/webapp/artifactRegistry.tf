#############################
# Artifact Registry Repository
#############################
resource "google_artifact_registry_repository" "repository" {
  repository_id = var.repository_name
  location      = var.region
  format        = "DOCKER"
  description   = "Artifact Registry for Cloud Run"
}
