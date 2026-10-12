FROM rust:1-slim AS builder
WORKDIR /build
COPY apps/api-rust/Cargo.toml apps/api-rust/Cargo.lock ./
COPY apps/api-rust/src ./src
RUN cargo build --release

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home astromap
COPY --from=builder /build/target/release/api-rust /usr/local/bin/astromap-api
USER astromap
EXPOSE 8080
CMD ["astromap-api"]
