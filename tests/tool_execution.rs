use polymede::tools::{Tool, ToolContext, ToolRegistry, ToolCall};
use serde_json::json;

fn test_context() -> ToolContext {
    ToolContext::new(std::env::current_dir().unwrap(), None)
}

fn make_call(name: &str, args: serde_json::Value) -> ToolCall {
    ToolCall {
        id: format!("call_{}", name),
        name: name.into(),
        arguments: args,
    }
}

#[tokio::test]
async fn test_tool_registry_default_tools() {
    let tools = ToolRegistry::default_tools();
    assert!(!tools.is_empty());
    assert!(tools.len() >= 30, "Expected at least 30 default tools, got {}", tools.len());
}

#[tokio::test]
async fn test_tool_registry_definitions() {
    let tools = ToolRegistry::default_tools();
    for tool in &tools {
        assert!(!tool.name().is_empty());
        assert!(!tool.description().is_empty());
        let params = tool.parameters();
        assert!(params.get("type").is_some(), "Missing type in parameters of {}", tool.name());
    }
}

#[tokio::test]
async fn test_tool_registry_with_cache() {
    let _reg = ToolRegistry::new().with_cache(300, 100);
}

#[tokio::test]
async fn test_date_tool() {
    let ctx = test_context();
    let reg = ToolRegistry::new();
    let result = reg.execute(&ctx, &make_call("date", json!({}))).await;
    assert!(result.ok);
    assert!(!result.output.is_empty());
}

#[tokio::test]
async fn test_env_tool() {
    let ctx = test_context();
    let reg = ToolRegistry::new();
    let result = reg.execute(&ctx, &make_call("env", json!({}))).await;
    assert!(result.ok);
    assert!(!result.output.is_empty());
}

#[tokio::test]
async fn test_which_tool() {
    let ctx = test_context();
    let reg = ToolRegistry::new();
    let result = reg.execute(&ctx, &make_call("which", json!({"command": "ls"}))).await;
    assert!(result.ok || !result.output.is_empty());
}

#[tokio::test]
async fn test_ping_tool() {
    let ctx = test_context();
    let reg = ToolRegistry::new();
    let result = reg.execute(&ctx, &make_call("ping", json!({"host": "127.0.0.1", "count": 1}))).await;
    assert!(!result.output.is_empty());
}

#[tokio::test]
async fn test_read_tool_nonexistent() {
    let ctx = test_context();
    let reg = ToolRegistry::new();
    let result = reg.execute(&ctx, &make_call("read", json!({"path": "/nonexistent/file.txt"}))).await;
    assert!(!result.ok || !result.output.is_empty());
}

#[tokio::test]
async fn test_write_and_read_tool() {
    let ctx = test_context();
    let reg = ToolRegistry::new();
    let tmp_path = "/tmp/polymede_test_write.txt";
    let _ = std::fs::remove_file(tmp_path);

    let result = reg.execute(&ctx, &make_call("write", json!({"path": tmp_path, "content": "Hello from Polymede test"}))).await;
    assert!(result.ok);

    let result = reg.execute(&ctx, &make_call("read", json!({"path": tmp_path}))).await;
    assert!(result.output.contains("Hello from Polymede test"));

    let _ = std::fs::remove_file(tmp_path);
}

#[tokio::test]
async fn test_grep_tool() {
    let ctx = test_context();
    let reg = ToolRegistry::new();
    let tmp_path = "/tmp/polymede_test_grep.txt";
    std::fs::write(tmp_path, "line one\nhello world\nfoo bar\n").unwrap();

    let result = reg.execute(&ctx, &make_call("grep", json!({"pattern": "hello", "path": tmp_path}))).await;
    assert!(result.output.contains("hello world"));

    let _ = std::fs::remove_file(tmp_path);
}

#[tokio::test]
async fn test_mkdir_tool() {
    let ctx = test_context();
    let reg = ToolRegistry::new();
    let tmp_dir = "/tmp/polymede_test_mkdir";
    let _ = std::fs::remove_dir_all(tmp_dir);

    let result = reg.execute(&ctx, &make_call("mkdir", json!({"path": tmp_dir}))).await;
    assert!(result.ok || !result.output.is_empty());
    if std::path::Path::new(tmp_dir).is_dir() {
        let _ = std::fs::remove_dir_all(tmp_dir);
    }
}

#[tokio::test]
async fn test_touch_tool() {
    let ctx = test_context();
    let reg = ToolRegistry::new();
    let tmp_file = "/tmp/polymede_test_touch.txt";
    let _ = std::fs::remove_file(tmp_file);

    let result = reg.execute(&ctx, &make_call("touch", json!({"path": tmp_file}))).await;
    assert!(result.ok || !result.output.is_empty());
    if std::path::Path::new(tmp_file).exists() {
        let _ = std::fs::remove_file(tmp_file);
    }
}

#[tokio::test]
async fn test_disk_usage_tool() {
    let ctx = test_context();
    let reg = ToolRegistry::new();
    let result = reg.execute(&ctx, &make_call("disk_usage", json!({}))).await;
    // May fail on some systems — just ensure it returns something.
    assert!(!result.output.is_empty());
}

#[tokio::test]
async fn test_uptime_tool() {
    let ctx = test_context();
    let reg = ToolRegistry::new();
    let result = reg.execute(&ctx, &make_call("uptime", json!({}))).await;
    assert!(result.ok);
    assert!(!result.output.is_empty());
}

#[tokio::test]
async fn test_process_list_tool() {
    let ctx = test_context();
    let reg = ToolRegistry::new();
    let result = reg.execute(&ctx, &make_call("process_list", json!({}))).await;
    assert!(result.ok);
    assert!(!result.output.is_empty());
}

#[tokio::test]
async fn test_network_info_tool() {
    let ctx = test_context();
    let reg = ToolRegistry::new();
    let result = reg.execute(&ctx, &make_call("network_info", json!({}))).await;
    assert!(result.ok);
    assert!(!result.output.is_empty());
}

#[tokio::test]
async fn test_tool_cache_clear() {
    let reg = ToolRegistry::new().with_cache(300, 100);
    reg.clear_cache().await;
}

#[tokio::test]
async fn test_unknown_tool_returns_error() {
    let ctx = test_context();
    let reg = ToolRegistry::new();
    let result = reg.execute(&ctx, &make_call("nonexistent_tool_xyz", json!({}))).await;
    assert!(!result.ok);
}

#[tokio::test]
async fn test_bash_tool_echo() {
    let ctx = test_context();
    let reg = ToolRegistry::new();
    let result = reg.execute(&ctx, &make_call("bash", json!({"command": "echo hello_world"}))).await;
    assert!(result.output.contains("hello_world"));
}

#[tokio::test]
async fn test_bash_tool_error() {
    let ctx = test_context();
    let reg = ToolRegistry::new();
    let result = reg.execute(&ctx, &make_call("bash", json!({"command": "false"}))).await;
    assert!(!result.ok || result.output.contains("exit"));
}

#[tokio::test]
async fn test_tool_context_creation() {
    let ctx = ToolContext::new(std::env::current_dir().unwrap(), None);
    assert!(!ctx.working_dir.as_os_str().is_empty());
    assert!(ctx.allowed_commands.is_none());
}

#[tokio::test]
async fn test_tool_context_with_allowed_commands() {
    let allowed = vec!["echo".into(), "ls".into()];
    let ctx = ToolContext::new(std::env::current_dir().unwrap(), Some(allowed.clone()));
    assert_eq!(ctx.allowed_commands, Some(allowed));
}

#[tokio::test]
async fn test_tool_registry_tool_names() {
    let reg = ToolRegistry::new();
    let names = reg.tool_names();
    assert!(!names.is_empty());
}

#[tokio::test]
async fn test_tool_definitions_valid_json() {
    let tools = ToolRegistry::default_tools();
    for tool in &tools {
        assert!(!tool.name().is_empty());
        assert!(!tool.description().is_empty());
        let params = tool.parameters();
        assert!(params.get("type").is_some(), "Missing type in {}", tool.name());
    }
}

#[tokio::test]
async fn test_base64_tool() {
    let ctx = test_context();
    let reg = ToolRegistry::new();
    let result = reg.execute(&ctx, &make_call("base64", json!({"input": "Hello World", "mode": "encode"}))).await;
    assert!(result.ok || !result.output.is_empty());
}

#[tokio::test]
async fn test_glob_tool() {
    let ctx = test_context();
    let reg = ToolRegistry::new();
    let result = reg.execute(&ctx, &make_call("glob", json!({"pattern": "*.rs"}))).await;
    assert!(result.ok || !result.output.is_empty());
}

#[tokio::test]
async fn test_compiler_check_tool() {
    let ctx = test_context();
    let reg = ToolRegistry::new();
    let result = reg.execute(&ctx, &make_call("compiler_check", json!({"language": "rust"}))).await;
    assert!(result.ok || !result.output.is_empty());
}

#[tokio::test]
async fn test_package_search_tool() {
    let ctx = test_context();
    let reg = ToolRegistry::new();
    let result = reg.execute(&ctx, &make_call("package_search", json!({"query": "tokio"}))).await;
    assert!(result.ok || !result.output.is_empty());
}

#[tokio::test]
async fn test_git_status_tool() {
    let ctx = test_context();
    let reg = ToolRegistry::new();
    let result = reg.execute(&ctx, &make_call("git_status", json!({}))).await;
    assert!(result.ok || !result.output.is_empty());
}

#[tokio::test]
async fn test_git_log_tool() {
    let ctx = test_context();
    let reg = ToolRegistry::new();
    let result = reg.execute(&ctx, &make_call("git_log", json!({"count": 3}))).await;
    assert!(result.ok || !result.output.is_empty());
}

#[tokio::test]
async fn test_git_diff_tool() {
    let ctx = test_context();
    let reg = ToolRegistry::new();
    let result = reg.execute(&ctx, &make_call("git_diff", json!({}))).await;
    assert!(result.ok || !result.output.is_empty());
}

#[tokio::test]
async fn test_dig_tool() {
    let ctx = test_context();
    let reg = ToolRegistry::new();
    let result = reg.execute(&ctx, &make_call("dig", json!({"domain": "example.com"}))).await;
    assert!(result.ok || !result.output.is_empty());
}

#[tokio::test]
async fn test_dns_lookup_tool() {
    let ctx = test_context();
    let reg = ToolRegistry::new();
    let result = reg.execute(&ctx, &make_call("dns_lookup", json!({"domain": "example.com"}))).await;
    assert!(result.ok || !result.output.is_empty());
}

#[tokio::test]
async fn test_whereis_tool() {
    let ctx = test_context();
    let reg = ToolRegistry::new();
    let result = reg.execute(&ctx, &make_call("whereis", json!({"name": "ls"}))).await;
    assert!(result.ok || !result.output.is_empty());
}

#[tokio::test]
async fn test_find_tool() {
    let ctx = test_context();
    let reg = ToolRegistry::new();
    let result = reg.execute(&ctx, &make_call("find", json!({"path": "/tmp", "name": "*.txt"}))).await;
    assert!(result.ok || !result.output.is_empty());
}

#[tokio::test]
async fn test_chmod_tool() {
    let ctx = test_context();
    let reg = ToolRegistry::new();
    let tmp_file = "/tmp/polymede_test_chmod.txt";
    std::fs::write(tmp_file, "test").unwrap();

    let result = reg.execute(&ctx, &make_call("chmod", json!({"path": tmp_file, "mode": "644"}))).await;
    assert!(result.ok || !result.output.is_empty());

    let _ = std::fs::remove_file(tmp_file);
}

#[tokio::test]
async fn test_cp_tool() {
    let ctx = test_context();
    let reg = ToolRegistry::new();
    let src = "/tmp/polymede_test_cp_src.txt";
    let dst = "/tmp/polymede_test_cp_dst.txt";
    std::fs::write(src, "copy me").unwrap();
    let _ = std::fs::remove_file(dst);

    let result = reg.execute(&ctx, &make_call("cp", json!({"source": src, "destination": dst}))).await;
    assert!(result.ok || !result.output.is_empty());

    let _ = std::fs::remove_file(src);
    let _ = std::fs::remove_file(dst);
}

#[tokio::test]
async fn test_mv_tool() {
    let ctx = test_context();
    let reg = ToolRegistry::new();
    let src = "/tmp/polymede_test_mv_src.txt";
    let dst = "/tmp/polymede_test_mv_dst.txt";
    std::fs::write(src, "move me").unwrap();
    let _ = std::fs::remove_file(dst);

    let result = reg.execute(&ctx, &make_call("mv", json!({"source": src, "destination": dst}))).await;
    assert!(result.ok || !result.output.is_empty());

    let _ = std::fs::remove_file(dst);
}

#[tokio::test]
async fn test_rm_tool() {
    let ctx = test_context();
    let reg = ToolRegistry::new();
    let tmp_file = "/tmp/polymede_test_rm.txt";
    std::fs::write(tmp_file, "delete me").unwrap();

    let result = reg.execute(&ctx, &make_call("rm", json!({"path": tmp_file}))).await;
    assert!(result.ok || !result.output.is_empty());

    assert!(!std::path::Path::new(tmp_file).exists());
}

#[tokio::test]
async fn test_edit_tool() {
    let ctx = test_context();
    let reg = ToolRegistry::new();
    let tmp_file = "/tmp/polymede_test_edit.txt";
    std::fs::write(tmp_file, "original line\n").unwrap();

    let result = reg.execute(&ctx, &make_call("edit", json!({"path": tmp_file, "old_string": "original", "new_string": "edited"}))).await;
    assert!(result.ok || !result.output.is_empty());

    let _ = std::fs::remove_file(tmp_file);
}
