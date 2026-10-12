use std::{env, sync::Arc, time::Duration};

use axum::{
    Json, Router,
    extract::{Path, Query, State},
    http::{Method, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tower_http::{
    cors::{Any, CorsLayer},
    trace::TraceLayer,
};

#[derive(Clone)]
pub struct AppState {
    inference_url: String,
    client: Client,
    targets: Arc<Vec<Target>>,
}

impl AppState {
    pub fn new(inference_url: impl Into<String>, client: Client, targets: Vec<Target>) -> Self {
        Self {
            inference_url: inference_url.into(),
            client,
            targets: Arc::new(targets),
        }
    }

    pub fn with_inference_url(inference_url: impl Into<String>) -> Self {
        Self {
            inference_url: inference_url.into(),
            client: Client::builder()
                .timeout(Duration::from_secs(120))
                .build()
                .expect("HTTP client configuration must be valid"),
            targets: Arc::new(seed_targets()),
        }
    }
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            inference_url: env::var("ASTROMAP_INFERENCE_URL")
                .unwrap_or_else(|_| "http://127.0.0.1:8001".to_owned()),
            client: Client::builder()
                .timeout(Duration::from_secs(120))
                .build()
                .expect("HTTP client configuration must be valid"),
            targets: Arc::new(seed_targets()),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Target {
    pub tic_id: u64,
    pub name: String,
    pub ra: f64,
    pub dec: f64,
    pub magnitude: f32,
    pub spectral_type: String,
    pub score: Option<f32>,
}

#[derive(Debug, Deserialize)]
struct SearchQuery {
    ra: Option<f64>,
    dec: Option<f64>,
    radius_deg: Option<f64>,
    min_score: Option<f32>,
}

#[derive(Debug, Deserialize)]
struct AnalysisQuery {
    sector: Option<u16>,
}

#[derive(Debug, Serialize)]
struct HealthResponse {
    status: &'static str,
    service: &'static str,
    version: &'static str,
}

#[derive(Debug, Serialize)]
struct TargetList {
    items: Vec<Target>,
    total: usize,
}

#[derive(Debug, Serialize)]
struct ApiErrorBody {
    code: &'static str,
    message: String,
}

struct ApiError {
    status: StatusCode,
    code: &'static str,
    message: String,
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (
            self.status,
            Json(ApiErrorBody {
                code: self.code,
                message: self.message,
            }),
        )
            .into_response()
    }
}

pub fn app(state: AppState) -> Router {
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods([Method::GET, Method::POST])
        .allow_headers(Any);

    Router::new()
        .route("/health/live", get(health))
        .route("/health/ready", get(health))
        .route("/api/v1/targets", get(search_targets))
        .route("/api/v1/targets/{tic_id}", get(get_target))
        .route(
            "/api/v1/targets/{tic_id}/observations",
            get(get_observations),
        )
        .route("/api/v1/analyses/{tic_id}", post(analyze_target))
        .layer(cors)
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

async fn health() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok",
        service: "api",
        version: env!("CARGO_PKG_VERSION"),
    })
}

async fn search_targets(
    State(state): State<AppState>,
    Query(query): Query<SearchQuery>,
) -> Result<Json<TargetList>, ApiError> {
    if query.ra.is_some() != query.dec.is_some() {
        return Err(ApiError {
            status: StatusCode::UNPROCESSABLE_ENTITY,
            code: "INVALID_COORDINATES",
            message: "ra and dec must be supplied together".to_owned(),
        });
    }
    let radius = query.radius_deg.unwrap_or(180.0);
    if !(0.0..=180.0).contains(&radius) {
        return Err(ApiError {
            status: StatusCode::UNPROCESSABLE_ENTITY,
            code: "INVALID_RADIUS",
            message: "radius_deg must be between 0 and 180".to_owned(),
        });
    }

    let items: Vec<Target> = state
        .targets
        .iter()
        .filter(|target| {
            let score_match = target.score.unwrap_or(0.0) >= query.min_score.unwrap_or(0.0);
            let position_match = match (query.ra, query.dec) {
                (Some(ra), Some(dec)) => {
                    angular_distance_deg(ra, dec, target.ra, target.dec) <= radius
                }
                _ => true,
            };
            score_match && position_match
        })
        .cloned()
        .collect();
    let total = items.len();
    Ok(Json(TargetList { items, total }))
}

async fn get_target(
    State(state): State<AppState>,
    Path(tic_id): Path<u64>,
) -> Result<Json<Target>, ApiError> {
    state
        .targets
        .iter()
        .find(|target| target.tic_id == tic_id)
        .cloned()
        .map(Json)
        .ok_or(ApiError {
            status: StatusCode::NOT_FOUND,
            code: "TARGET_NOT_FOUND",
            message: format!("TIC {tic_id} is not present in the local catalog"),
        })
}

async fn analyze_target(
    State(state): State<AppState>,
    Path(tic_id): Path<u64>,
    Query(query): Query<AnalysisQuery>,
) -> Result<Json<Value>, ApiError> {
    if tic_id == 0 {
        return Err(ApiError {
            status: StatusCode::UNPROCESSABLE_ENTITY,
            code: "INVALID_TIC_ID",
            message: "tic_id must be positive".to_owned(),
        });
    }
    let url = inference_url(&state.inference_url, tic_id, query.sector);
    let response = state
        .client
        .post(url)
        .send()
        .await
        .map_err(|error| ApiError {
            status: StatusCode::SERVICE_UNAVAILABLE,
            code: "INFERENCE_UNAVAILABLE",
            message: format!("inference service unavailable: {error}"),
        })?;
    if !response.status().is_success() {
        return Err(ApiError {
            status: StatusCode::BAD_GATEWAY,
            code: "INFERENCE_FAILED",
            message: format!("inference service returned {}", response.status()),
        });
    }
    let prediction = response.json::<Value>().await.map_err(|error| ApiError {
        status: StatusCode::BAD_GATEWAY,
        code: "INVALID_INFERENCE_RESPONSE",
        message: error.to_string(),
    })?;
    Ok(Json(
        json!({ "status": "completed", "prediction": prediction }),
    ))
}

async fn get_observations(
    State(state): State<AppState>,
    Path(tic_id): Path<u64>,
) -> Result<Json<Value>, ApiError> {
    if tic_id == 0 {
        return Err(ApiError {
            status: StatusCode::UNPROCESSABLE_ENTITY,
            code: "INVALID_TIC_ID",
            message: "tic_id must be positive".to_owned(),
        });
    }
    let url = format!(
        "{}/v1/observations/{tic_id}",
        state.inference_url.trim_end_matches('/')
    );
    let response = state
        .client
        .get(url)
        .send()
        .await
        .map_err(|error| ApiError {
            status: StatusCode::SERVICE_UNAVAILABLE,
            code: "MAST_UNAVAILABLE",
            message: format!("observation service unavailable: {error}"),
        })?;
    if !response.status().is_success() {
        return Err(ApiError {
            status: StatusCode::BAD_GATEWAY,
            code: "OBSERVATION_QUERY_FAILED",
            message: format!("observation service returned {}", response.status()),
        });
    }
    response
        .json::<Value>()
        .await
        .map(Json)
        .map_err(|error| ApiError {
            status: StatusCode::BAD_GATEWAY,
            code: "INVALID_OBSERVATION_RESPONSE",
            message: error.to_string(),
        })
}

fn inference_url(base: &str, tic_id: u64, sector: Option<u16>) -> String {
    let base = base.trim_end_matches('/');
    match sector {
        Some(sector) => format!("{base}/v1/infer/{tic_id}?sector={sector}"),
        None => format!("{base}/v1/infer/{tic_id}"),
    }
}

fn angular_distance_deg(ra1: f64, dec1: f64, ra2: f64, dec2: f64) -> f64 {
    let (ra1, dec1, ra2, dec2) = (
        ra1.to_radians(),
        dec1.to_radians(),
        ra2.to_radians(),
        dec2.to_radians(),
    );
    let cosine = dec1.sin() * dec2.sin() + dec1.cos() * dec2.cos() * (ra1 - ra2).cos();
    cosine.clamp(-1.0, 1.0).acos().to_degrees()
}

fn seed_targets() -> Vec<Target> {
    vec![
        Target {
            tic_id: 150428135,
            name: "TOI 700".to_owned(),
            ra: 97.0967865903209,
            dec: -65.5793115481931,
            magnitude: 10.9102,
            spectral_type: "M2V".to_owned(),
            score: None,
        },
        Target {
            tic_id: 307210830,
            name: "TIC 307210830".to_owned(),
            ra: 124.531756290083,
            dec: -68.3129998725044,
            magnitude: 9.41242,
            spectral_type: "TIC star".to_owned(),
            score: None,
        },
        Target {
            tic_id: 261136679,
            name: "TIC 261136679".to_owned(),
            ra: 84.2911879979852,
            dec: -80.4691197969941,
            magnitude: 5.1054,
            spectral_type: "TIC star".to_owned(),
            score: None,
        },
    ]
}

#[cfg(test)]
mod tests {
    use axum::{
        body::Body,
        http::{Request, StatusCode},
    };
    use tower::ServiceExt;

    use super::{AppState, angular_distance_deg, app, inference_url};

    #[tokio::test]
    async fn health_endpoint_is_ready() {
        let response = app(AppState::default())
            .oneshot(
                Request::builder()
                    .uri("/health/ready")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn target_search_filters_by_score() {
        let response = app(AppState::default())
            .oneshot(
                Request::builder()
                    .uri("/api/v1/targets?min_score=0.9")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[test]
    fn angular_distance_handles_identical_coordinates() {
        assert!(angular_distance_deg(95.0, -65.0, 95.0, -65.0) < 1e-6);
    }

    #[test]
    fn selected_sector_is_forwarded_to_inference() {
        assert_eq!(
            inference_url("http://inference:8001/", 150428135, Some(3)),
            "http://inference:8001/v1/infer/150428135?sector=3"
        );
    }

    #[tokio::test]
    async fn analyses_endpoint_proxies_prediction() {
        use axum::routing::post;
        use serde_json::json;

        let mock_app = axum::Router::new().route(
            "/v1/infer/{tic_id}",
            post(|| async {
                axum::Json(json!({
                    "probability": 0.95,
                    "label": "candidate",
                    "depth": 0.0012,
                    "duration_days": 0.15,
                    "period_days": 3.5,
                    "model_version": "bls-screening-0.1.0",
                    "score_kind": "bls_screening_score",
                    "data_source": "mast:TESS/product/test.fits",
                    "mission": "TESS/SPOC",
                    "sector": 3,
                    "cadence_count": 512,
                    "time": [0.0, 1.0],
                    "normalized_flux": [1.0, 0.99]
                }))
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        tokio::spawn(async move {
            let _ = axum::serve(listener, mock_app).await;
        });

        let app_state = AppState::with_inference_url(format!("http://127.0.0.1:{port}"));
        let response = app(app_state)
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/v1/analyses/150428135?sector=3")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let json_val: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json_val["status"], "completed");
        assert_eq!(json_val["prediction"]["label"], "candidate");
        assert_eq!(json_val["prediction"]["sector"], 3);
    }

    #[tokio::test]
    async fn observations_endpoint_proxies_observation_list() {
        use axum::routing::get;
        use serde_json::json;

        let mock_app = axum::Router::new().route(
            "/v1/observations/{tic_id}",
            get(|| async {
                axum::Json(json!({
                    "items": [
                        {
                            "sector": 3,
                            "mission": "TESS",
                            "year": 2018,
                            "exposure_seconds": 120.0,
                            "source_uri": "mast:TESS/product/test.fits"
                        }
                    ],
                    "total": 1
                }))
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        tokio::spawn(async move {
            let _ = axum::serve(listener, mock_app).await;
        });

        let app_state = AppState::with_inference_url(format!("http://127.0.0.1:{port}"));
        let response = app(app_state)
            .oneshot(
                Request::builder()
                    .uri("/api/v1/targets/150428135/observations")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let json_val: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json_val["total"], 1);
        assert_eq!(json_val["items"][0]["sector"], 3);
    }

    #[tokio::test]
    async fn analyses_returns_502_when_downstream_fails() {
        use axum::routing::post;

        let mock_app = axum::Router::new().route(
            "/v1/infer/{tic_id}",
            post(|| async { (StatusCode::INTERNAL_SERVER_ERROR, "error") }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        tokio::spawn(async move {
            let _ = axum::serve(listener, mock_app).await;
        });

        let app_state = AppState::with_inference_url(format!("http://127.0.0.1:{port}"));
        let response = app(app_state)
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/v1/analyses/150428135")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    }

    #[tokio::test]
    async fn observations_returns_503_when_downstream_unreachable() {
        // Port 1 is reserved and normally unreachable
        let app_state = AppState::with_inference_url("http://127.0.0.1:1");
        let response = app(app_state)
            .oneshot(
                Request::builder()
                    .uri("/api/v1/targets/150428135/observations")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    }
}
