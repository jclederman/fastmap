// SPDX-License-Identifier: AGPL-3.0-or-later
//! Build tasks for FastMap.
//!
//! ```text
//! cargo xtask web [--serve] [--port N] [--check-size] [--dev]
//! ```
//!
//! `web` builds the two wasm variants (`webgpu` and `webgl`), runs `wasm-bindgen` and, if it
//! can be found, `wasm-opt`, then assembles `dist/` from `web/`. `--serve` serves `dist/`
//! locally. `--check-size` fails if a compressed binary exceeds its budget, which CI uses.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Command, exit};

/// Gzip size budgets in KiB, per variant. These are a CI guard against size regressions.
const BUDGET_KIB: [(&str, u64); 2] = [("webgpu", 300), ("webgl", 1100)];

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().to_path_buf()
}

fn run(cmd: &mut Command) {
    eprintln!("$ {:?}", cmd);
    let status = cmd.status().unwrap_or_else(|e| panic!("failed to run {:?}: {e}", cmd.get_program()));
    if !status.success() {
        exit(status.code().unwrap_or(1));
    }
}

fn find_wasm_opt() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("WASM_OPT") {
        return Some(PathBuf::from(p));
    }
    let ok = Command::new("wasm-opt").arg("--version").output().map(|o| o.status.success()).unwrap_or(false);
    ok.then(|| PathBuf::from("wasm-opt"))
}

fn gzip_len(data: &[u8]) -> u64 {
    let mut e = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::best());
    e.write_all(data).unwrap();
    e.finish().unwrap().len() as u64
}

/// FNV-1a: a tiny content hash used for cache busting.
fn hash(data: &[u8]) -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in data {
        h = (h ^ *b as u64).wrapping_mul(0x0100_0000_01b3);
    }
    format!("{:08x}", h as u32)
}

fn build_web(dev: bool, check_size: bool) {
    let root = root();
    let dist = root.join("dist");
    let _ = std::fs::remove_dir_all(&dist);
    std::fs::create_dir_all(&dist).unwrap();
    let wasm_opt = find_wasm_opt();
    if wasm_opt.is_none() && !dev {
        eprintln!("warning: wasm-opt not found (set WASM_OPT or put it on PATH); binaries will be larger");
    }
    let mut over_budget = false;
    let mut versions = String::new();
    for variant in ["webgpu", "webgl"] {
        let target_dir = root.join("target").join(format!("web-{variant}"));
        let mut cargo = Command::new(env!("CARGO"));
        cargo.current_dir(&root).args(["build", "-p", "fastmap-web", "--target", "wasm32-unknown-unknown"]);
        cargo.args(["--no-default-features", "--features", variant, "--target-dir"]).arg(&target_dir);
        if !dev {
            cargo.arg("--release");
        }
        run(&mut cargo);
        let profile = if dev { "debug" } else { "release" };
        let wasm = target_dir.join("wasm32-unknown-unknown").join(profile).join("fastmap_web.wasm");
        let name = format!("fastmap_{variant}");
        run(Command::new("wasm-bindgen")
            .args(["--target", "web", "--no-typescript", "--out-name", &name, "--out-dir"])
            .arg(&dist)
            .arg(&wasm));
        let out = dist.join(format!("{name}_bg.wasm"));
        if let (Some(opt), false) = (&wasm_opt, dev) {
            run(Command::new(opt)
                .arg("-O3")
                .args([
                    "--strip-debug",
                    "--strip-producers",
                    "--enable-simd",
                    "--enable-bulk-memory",
                    "--enable-nontrapping-float-to-int",
                    "--enable-sign-ext",
                    "--enable-mutable-globals",
                    "--enable-reference-types",
                    "--enable-multivalue",
                    "-o",
                ])
                .arg(&out)
                .arg(&out));
        }
        let bytes = std::fs::read(&out).unwrap();
        let gz = gzip_len(&bytes);
        versions += &format!("{variant}:{} ", hash(&bytes));
        eprintln!("{name}_bg.wasm: {} KiB raw, {} KiB gzip", bytes.len() / 1024, gz / 1024);
        let budget = BUDGET_KIB.iter().find(|b| b.0 == variant).unwrap().1;
        if check_size && gz / 1024 > budget {
            eprintln!("error: {name} is {} KiB gzipped, over its {budget} KiB budget", gz / 1024);
            over_budget = true;
        }
    }
    // Copy static files, stamping the build hash into the loader for cache busting.
    let stamp = hash(versions.as_bytes());
    for entry in std::fs::read_dir(root.join("web")).unwrap() {
        let p = entry.unwrap().path();
        let dst = dist.join(p.file_name().unwrap());
        if p.extension().is_some_and(|e| e == "html") {
            let html = std::fs::read_to_string(&p).unwrap().replace("__BUILD__", &stamp);
            std::fs::write(dst, html).unwrap();
        } else {
            std::fs::copy(&p, dst).unwrap();
        }
    }
    eprintln!("built dist/ (build {stamp})");
    if over_budget {
        exit(1);
    }
}

fn mime(path: &Path) -> &'static str {
    match path.extension().and_then(|e| e.to_str()).unwrap_or("") {
        "html" => "text/html; charset=utf-8",
        "js" => "text/javascript",
        "wasm" => "application/wasm",
        "css" => "text/css",
        "svg" => "image/svg+xml",
        "png" => "image/png",
        "ico" => "image/x-icon",
        _ => "application/octet-stream",
    }
}

fn handle(mut stream: TcpStream, dist: &Path) {
    let mut line = String::new();
    if BufReader::new(&stream).read_line(&mut line).is_err() {
        return;
    }
    let path = line.split_whitespace().nth(1).unwrap_or("/");
    let path = path.split(['?', '#']).next().unwrap_or("/");
    let rel = if path == "/" { "index.html" } else { path.trim_start_matches('/') };
    let file = dist.join(rel);
    let ok = !rel.contains("..") && file.is_file();
    let (status, body, ty) = if ok {
        let mut b = Vec::new();
        std::fs::File::open(&file).and_then(|mut f| f.read_to_end(&mut b)).unwrap();
        ("200 OK", b, mime(&file))
    } else {
        ("404 Not Found", b"not found".to_vec(), "text/plain")
    };
    let head = format!(
        "HTTP/1.1 {status}\r\nContent-Type: {ty}\r\nContent-Length: {}\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n",
        body.len()
    );
    let _ = stream.write_all(head.as_bytes()).and_then(|_| stream.write_all(&body));
}

fn serve(port: u16) {
    let dist = root().join("dist");
    let listener = TcpListener::bind(("127.0.0.1", port)).unwrap_or_else(|e| panic!("bind :{port}: {e}"));
    eprintln!("serving dist/ at http://localhost:{port}/");
    for stream in listener.incoming().flatten() {
        let dist = dist.clone();
        std::thread::spawn(move || handle(stream, &dist));
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let has = |f: &str| args.iter().any(|a| a == f);
    let port = args
        .iter()
        .position(|a| a == "--port")
        .and_then(|i| args.get(i + 1))
        .and_then(|p| p.parse().ok())
        .unwrap_or(8080);
    match args.first().map(String::as_str) {
        Some("web") => {
            build_web(has("--dev"), has("--check-size"));
            if has("--serve") {
                serve(port);
            }
        }
        Some("serve") => serve(port),
        _ => {
            eprintln!(
                "usage: cargo xtask web [--serve] [--port N] [--check-size] [--dev]\n       cargo xtask serve [--port N]"
            );
            exit(2);
        }
    }
}
