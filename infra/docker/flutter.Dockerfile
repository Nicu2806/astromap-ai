FROM ghcr.io/cirruslabs/flutter:stable AS builder
ARG ASTROMAP_API_URL=http://localhost:8080
WORKDIR /app
COPY apps/client-flutter/pubspec.yaml apps/client-flutter/pubspec.lock ./
RUN flutter pub get
COPY apps/client-flutter .
RUN flutter build web --release --dart-define=ASTROMAP_API_URL=${ASTROMAP_API_URL}

FROM nginx:1.27-alpine
COPY infra/docker/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=builder /app/build/web /usr/share/nginx/html
EXPOSE 80
