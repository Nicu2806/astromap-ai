use std::{env, net::SocketAddr};

use api_rust::{AppState, app};
use tokio::net::TcpListener;
use tracing::info;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();
    let port = env::var("ASTROMAP_API_PORT")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(8080);
    let address = SocketAddr::from(([0, 0, 0, 0], port));
    let listener = TcpListener::bind(address)
        .await
        .expect("API port must be available");
    info!(%address, "AstroMap-AI API listening");
    axum::serve(listener, app(AppState::default()))
        .await
        .expect("API server failed");
}
