use polymede::llm::{Message, MessageRole};

#[test]
fn print_message_payload() {
    let messages = vec![
        Message {
            role: MessageRole::System,
            content: "You are a helpful assistant.".into(),
            tool_calls: None,
            tool_call_id: None,
        },
        Message {
            role: MessageRole::User,
            content: "Hello".into(),
            tool_calls: None,
            tool_call_id: None,
        },
    ];

    let json = serde_json::to_string_pretty(&messages).unwrap();
    println!("Messages payload:\n{}", json);
}

#[test]
fn print_full_body() {
    use polymede::llm::{LlmClient, LlmConfig};

    let config = LlmConfig {
        provider: "custom".into(),
        model: "qwen3.6-27b-mtp".into(),
        api_key: Some("test".into()),
        base_url: Some("http://192.168.2.22:1234/v1".into()),
        fallback: None,
    };

    let messages = vec![
        Message {
            role: MessageRole::System,
            content: "You are a helpful assistant.".into(),
            tool_calls: None,
            tool_call_id: None,
        },
        Message {
            role: MessageRole::User,
            content: "Hello".into(),
            tool_calls: None,
            tool_call_id: None,
        },
    ];

    let client = LlmClient::new(0.7, None);
    // Access the private method via a workaround - just serialize manually
    let mut body = serde_json::Map::new();
    body.insert("model".into(), serde_json::Value::String(config.model.clone()));
    body.insert(
        "messages".into(),
        serde_json::to_value(&messages).unwrap(),
    );
    body.insert(
        "temperature".into(),
        serde_json::Value::Number(serde_json::Number::from_f64(0.7).unwrap()),
    );

    let json = serde_json::to_string_pretty(&body).unwrap();
    println!("\nFull request body:\n{}", json);
}
