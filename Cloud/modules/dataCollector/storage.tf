#############################
# Storage bucket for the data retrieval function code
#############################
resource "google_storage_bucket" "function_code_bucket" {
  name                        = var.bucket_name
  location                    = var.region
  uniform_bucket_level_access = true
}
resource "google_storage_bucket_object" "function_code" {
  name   = "function-${filemd5("${path.module}/../../scripts/dataColCF.zip")}.zip"
  bucket = google_storage_bucket.function_code_bucket.name
  source = "${path.module}/../../scripts/dataColCF.zip"
}
