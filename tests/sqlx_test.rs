use std::path::PathBuf;

#[tokio::test]
async fn test_sqlite_connect() {
    let state_dir = PathBuf::from("/tmp/polymede-test-state");
    println!("state_dir: {:?}", state_dir);
    
    std::fs::create_dir_all(&state_dir).expect("mkdir failed");
    
    let db_path = state_dir.join("test.db");
    println!("db_path: {:?}", db_path);
    
    // Exact same format as memory.rs connect()
    let url = format!("sqlite:{}", db_path.display());
    println!("url: {}", url);
    
    match sqlx::SqlitePool::connect(&url).await {
        Ok(pool) => {
            println!("CONNECTED OK");
            sqlx::query("CREATE TABLE IF NOT EXISTS test (id INTEGER)")
                .execute(&pool)
                .await
                .expect("create table failed");
            println!("TABLE CREATED OK");
        }
        Err(e) => {
            panic!("FAILED: {}", e);
        }
    }
    
    // Clean up
    let _ = std::fs::remove_file(&db_path);
    let _ = std::fs::remove_dir(&state_dir);
}

#[tokio::test]
async fn test_sqlite_connect_real_state() {
    use polymede::config::Config;
    
    let state_dir = Config::state_dir();
    println!("real state_dir: {:?}", state_dir);
    
    std::fs::create_dir_all(&state_dir).expect("mkdir failed");
    
    let db_path = state_dir.join("test.db");
    println!("db_path: {:?}", db_path);
    
    // Ensure parent exists (this is what connect() does)
    if let Some(parent) = db_path.parent() {
        std::fs::create_dir_all(parent).expect("parent mkdir failed");
        println!("parent {:?} exists: {}", parent, parent.exists());
    }
    
    let url = format!("sqlite:{}", db_path.display());
    println!("url: {}", url);
    
    match sqlx::SqlitePool::connect(&url).await {
        Ok(pool) => {
            println!("CONNECTED OK");
            sqlx::query("CREATE TABLE IF NOT EXISTS test (id INTEGER)")
                .execute(&pool)
                .await
                .expect("create table failed");
            println!("TABLE CREATED OK");
        }
        Err(e) => {
            panic!("FAILED: {}", e);
        }
    }
    
    // Clean up
    let _ = std::fs::remove_file(&db_path);
}
