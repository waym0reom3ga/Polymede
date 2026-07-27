use polymede::llm::{Message, MessageRole};

#[test]
fn show_serialized_messages() {
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
    eprintln!("=== Serialized messages ===\n{}", json);
}
