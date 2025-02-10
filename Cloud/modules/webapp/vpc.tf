# resource "google_compute_network" "private_vpc" {
#   name                    = "private-cloudrun-vpc"
#   auto_create_subnetworks = false
# }

# resource "google_compute_subnetwork" "private_subnet" {
#   name                     = "private-cloudrun-subnet"
#   ip_cidr_range            = "10.8.0.0/24"
#   region                   = var.region
#   network                  = google_compute_network.private_vpc.id
#   private_ip_google_access = true
# }

# resource "google_vpc_access_connector" "cloudrun_connector" {
#   name          = "cloudrun-vpc-connector"
#   network       = google_compute_network.private_vpc.name
#   region        = var.region
#   ip_cidr_range = "10.8.1.0/28"
#   max_instances = 3
#   min_instances = 2
# }
