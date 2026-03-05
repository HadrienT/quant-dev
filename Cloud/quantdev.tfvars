project_id         = "quant-dev-442615"
project_number     = "956975863647"
region             = "europe-west1"
zone               = "europe-west1-b"
tf_service_account = "sa-qdev-tf-dev@quant-dev-442615.iam.gserviceaccount.com"

repository_name = "quant-modeling"
service_name    = "quantmodeling-web"

github_owner            = "HadrienT"
github_repo_name        = "quant-modeling"
builder_service_account = "sa-cloud-build@quant-dev-442615.iam.gserviceaccount.com"

domain_name = "tramonihadrien.com"

enable_webapp = true

enable_genealogy           = true
genealogy_repository_name  = "genealogy-api"
genealogy_service_name     = "genealogy-api"
genealogy_github_repo_name = "genealogy"

bq_reader_key_path = "./secrets/bq-key.json"
