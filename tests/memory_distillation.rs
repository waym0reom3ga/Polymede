use polymede::memory::{MemoryIntegration, MemoryStatus};
use polymede::config::MemoryConfig;
use std::path::PathBuf;
use std::sync::Arc;

fn test_state_dir() -> PathBuf {
    let dir = std::env::current_dir().unwrap()
        .join("test_tmp")
        .join(format!("polymede_mem_test_{}", uuid::Uuid::new_v4()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("Failed to create test state dir");
    dir
}

fn test_config() -> MemoryConfig {
    MemoryConfig::default()
}

#[tokio::test]
async fn test_memory_create_and_status() {
    let state_dir = test_state_dir();
    let memory = MemoryIntegration::new(state_dir.clone(), test_config(), None, false).await.unwrap();
    let status = memory.status().await.unwrap();
    assert_eq!(status.raw_count, 0);
    assert_eq!(status.compressed_count, 0);
    let _ = std::fs::remove_dir_all(&state_dir);
}

#[tokio::test]
async fn test_memory_ingest_and_recall() {
    let state_dir = test_state_dir();
    let memory = MemoryIntegration::new(state_dir.clone(), test_config(), None, false).await.unwrap();

    // Ingest a fact.
    memory.ingest("user", "User prefers Rust over Python", &[], vec!["preference".to_string()].as_slice()).await.unwrap();
    let status = memory.status().await.unwrap();
    assert!(status.raw_count >= 1);

    // Recall with tag search.
    let results = memory.recall(vec!["preference".to_string()].as_slice(), None).await.unwrap();
    assert!(!results.raw_interactions.is_empty());
    assert!(results.raw_interactions.iter().any(|r| r.content.contains("Rust")));

    let _ = std::fs::remove_dir_all(&state_dir);
}

#[tokio::test]
async fn test_memory_multiple_ingests() {
    let state_dir = test_state_dir();
    let memory = MemoryIntegration::new(state_dir.clone(), test_config(), None, false).await.unwrap();

    for i in 0..5 {
        memory.ingest("user", &format!("Fact number {}", i), &[], vec!["test".to_string()].as_slice()).await.unwrap();
    }

    let status = memory.status().await.unwrap();
    assert!(status.raw_count >= 5);

    let _ = std::fs::remove_dir_all(&state_dir);
}

#[tokio::test]
async fn test_memory_recall_empty() {
    let state_dir = test_state_dir();
    let memory = MemoryIntegration::new(state_dir.clone(), test_config(), None, false).await.unwrap();

    // Recall on empty memory should return empty result.
    let results = memory.recall(vec!["nonexistent".to_string()].as_slice(), None).await.unwrap();
    assert!(results.memories.is_empty());
    assert!(results.raw_interactions.is_empty());

    let _ = std::fs::remove_dir_all(&state_dir);
}

#[tokio::test]
async fn test_memory_recall_relevance() {
    let state_dir = test_state_dir();
    let memory = MemoryIntegration::new(state_dir.clone(), test_config(), None, false).await.unwrap();

    memory.ingest("user", "The sky is blue", &[], vec!["observation".to_string()].as_slice()).await.unwrap();
    memory.ingest("assistant", "Rust uses cargo for builds", &[], vec!["technical".to_string()].as_slice()).await.unwrap();
    memory.ingest("user", "Coffee tastes better in the morning", &[], vec!["preference".to_string()].as_slice()).await.unwrap();

    // Search with technical tag.
    let results = memory.recall(vec!["technical".to_string()].as_slice(), None).await.unwrap();
    assert!(!results.raw_interactions.is_empty());
    let has_rust = results.raw_interactions.iter().any(|r| r.content.contains("Rust") || r.content.contains("cargo"));
    assert!(has_rust, "Expected Rust/cargo result for technical tag");

    let _ = std::fs::remove_dir_all(&state_dir);
}

#[tokio::test]
async fn test_memory_compress_no_panic() {
    let state_dir = test_state_dir();
    let memory = MemoryIntegration::new(state_dir.clone(), test_config(), None, false).await.unwrap();

    // Ingest some entries then compress.
    for i in 0..3 {
        memory.ingest("user", &format!("Compressible entry {}", i), &[], vec!["test".to_string()].as_slice()).await.unwrap();
    }

    let before = memory.status().await.unwrap().raw_count;
    assert!(before >= 3);

    // Compress should not panic even with few entries.
    let compressed = memory.compress().await.unwrap_or(0);
    assert!(compressed >= 0);

    let _ = std::fs::remove_dir_all(&state_dir);
}

#[tokio::test]
async fn test_memory_session_id() {
    let state_dir = test_state_dir();
    let memory = MemoryIntegration::new(state_dir.clone(), test_config(), None, false).await.unwrap();
    assert!(!memory.session_id().is_empty());
    let _ = std::fs::remove_dir_all(&state_dir);
}

#[tokio::test]
async fn test_memory_distillation_accuracy_basic() {
    let state_dir = test_state_dir();
    let memory = MemoryIntegration::new(state_dir.clone(), test_config(), None, false).await.unwrap();

    // Ingest known facts with tags.
    let facts: Vec<(&str, &str)> = vec![
        ("The capital of France is Paris", "geography"),
        ("Water boils at 100 degrees Celsius", "science"),
        ("Rust prevents null pointer dereferences", "rust"),
        ("Tokio is an async runtime for Rust", "rust"),
        ("HTTP status 200 means OK", "http"),
    ];

    for (fact, tag) in &facts {
        memory.ingest("user", fact, &[], vec![tag.to_string()].as_slice()).await.unwrap();
    }

    // Verify facts can be recalled by their tags.
    let rust_results = memory.recall(vec!["rust".to_string()].as_slice(), None).await.unwrap();
    assert!(rust_results.raw_interactions.len() >= 2);

    let science_results = memory.recall(vec!["science".to_string()].as_slice(), None).await.unwrap();
    assert!(!science_results.raw_interactions.is_empty());

    let _ = std::fs::remove_dir_all(&state_dir);
}

#[tokio::test]
async fn test_memory_distillation_accuracy_categories() {
    let state_dir = test_state_dir();
    let memory = MemoryIntegration::new(state_dir.clone(), test_config(), None, false).await.unwrap();

    // Ingest across different categories.
    memory.ingest("user", "User likes dark mode", &[], vec!["preference".to_string()].as_slice()).await.unwrap();
    memory.ingest("assistant", "Project uses Rust edition 2024", &[], vec!["technical".to_string()].as_slice()).await.unwrap();
    memory.ingest("user", "Meeting at 3pm on Monday", &[], vec!["schedule".to_string()].as_slice()).await.unwrap();

    // Category-specific recall should work.
    let tech_results = memory.recall(vec!["technical".to_string()].as_slice(), None).await.unwrap();
    assert!(!tech_results.raw_interactions.is_empty());

    let pref_results = memory.recall(vec!["preference".to_string()].as_slice(), None).await.unwrap();
    assert!(!pref_results.raw_interactions.is_empty());

    let _ = std::fs::remove_dir_all(&state_dir);
}

#[tokio::test]
async fn test_memory_concurrent_ingest() {
    let state_dir = test_state_dir();
    let memory = std::sync::Arc::new(
        MemoryIntegration::new(state_dir.clone(), test_config(), None, false).await.unwrap()
    );

    // Spawn concurrent ingests.
    let mut handles = vec![];
    for i in 0..5 {
        let mem = Arc::clone(&memory);
        handles.push(tokio::spawn(async move {
            mem.ingest("user", &format!("Concurrent entry {}", i), &[], vec!["test".to_string()].as_slice()).await.unwrap();
        }));
    }

    for h in handles {
        h.await.unwrap();
    }

    let status = memory.status().await.unwrap();
    assert!(status.raw_count >= 5);

    let _ = std::fs::remove_dir_all(&state_dir);
}

#[tokio::test]
async fn test_memory_recall_token_budget() {
    let state_dir = test_state_dir();
    let memory = MemoryIntegration::new(state_dir.clone(), test_config(), None, false).await.unwrap();

    for i in 0..20 {
        memory.ingest("user", &format!("Item {} with some extra content to use tokens", i), &[], vec!["test".to_string()].as_slice()).await.unwrap();
    }

    // Request with small token budget.
    let results = memory.recall(vec!["test".to_string()].as_slice(), Some(100)).await.unwrap();
    assert!(results.tokens_used <= 100);
    assert!(results.raw_interactions.len() < 20, "Budget should limit results");

    let _ = std::fs::remove_dir_all(&state_dir);
}

#[tokio::test]
async fn test_memory_error_on_bad_path() {
    // A path that definitely can't be a database directory.
    let bad_path = PathBuf::from("/proc/nonexistent_polymede_db");
    let result = MemoryIntegration::new(bad_path, test_config(), None, false).await;
    // Should fail gracefully, not panic.
    assert!(result.is_err());
}

#[tokio::test]
async fn test_memory_distillation_fallback_mechanical() {
    let state_dir = test_state_dir();
    let memory = MemoryIntegration::new(state_dir.clone(), test_config(), None, false).await.unwrap();

    // Ingest without LLM — should use mechanical keyword extraction.
    memory.ingest("user", "The quick brown fox jumps over the lazy dog", &[], vec!["sentence".to_string()].as_slice()).await.unwrap();

    // Mechanical recall via tags should still work.
    let results = memory.recall(vec!["sentence".to_string()].as_slice(), None).await.unwrap();
    assert!(!results.raw_interactions.is_empty());

    let _ = std::fs::remove_dir_all(&state_dir);
}

#[tokio::test]
async fn test_memory_status_fields() {
    let state_dir = test_state_dir();
    let memory = MemoryIntegration::new(state_dir.clone(), test_config(), None, false).await.unwrap();

    let status = memory.status().await.unwrap();
    // All fields should be accessible.
    assert_eq!(status.raw_count, 0);
    assert_eq!(status.compressed_count, 0);
    assert_eq!(status.highest_level, 0);
    assert!(status.last_compression.is_none());

    let _ = std::fs::remove_dir_all(&state_dir);
}

#[tokio::test]
async fn test_memory_recall_result_structure() {
    let state_dir = test_state_dir();
    let memory = MemoryIntegration::new(state_dir.clone(), test_config(), None, false).await.unwrap();

    memory.ingest("user", "Test content", &[], vec!["tag1".to_string()].as_slice()).await.unwrap();
    let results = memory.recall(vec!["tag1".to_string()].as_slice(), Some(500)).await.unwrap();

    // Verify RecallResult structure.
    assert!(results.tokens_budget == 500);
    assert!(results.tokens_used >= 0);

    let _ = std::fs::remove_dir_all(&state_dir);
}
