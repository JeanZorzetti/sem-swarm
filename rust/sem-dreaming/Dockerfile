# Dockerfile for SEM-Swarm Dreaming Loop (Rust)
# This uses a multi-stage build to compile Rust into a tiny Debian image.

# Stage 1: Build
FROM rust:1.80 as builder

WORKDIR /usr/src/app

# We only want to build the dreaming loop, but we need the workspace Cargo.toml
# and the other crates if they exist.
# A simple way for a workspace is to copy everything and then build the specific package.
COPY . .

# Build the dreaming loop binary for release
# We build it statically if possible, but Debian bookworm-slim is fine for dynamic linking.
RUN cargo build --release --package sem-dreaming

# Stage 2: Run
FROM debian:bookworm-slim

WORKDIR /app

# Install OpenSSL/CA certificates needed by reqwest/native-tls
RUN apt-get update && apt-get install -y ca-certificates libssl3 && rm -rf /var/lib/apt/lists/*

# Copy the compiled binary from the builder stage
COPY --from=builder /usr/src/app/rust/target/release/sem-dreaming /usr/local/bin/sem-dreaming

# The Dreaming Loop is designed to run periodically (e.g. overnight).
# Since Easypanel Workers restart if the command exits, we will wrap it in a while loop.
# It will run, sleep for 3 hours (10800 seconds), and run again.
CMD ["sh", "-c", "while true; do sem-dreaming; echo 'Sleeping for 3 hours...'; sleep 10800; done"]
