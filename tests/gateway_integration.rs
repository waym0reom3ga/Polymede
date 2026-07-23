use polymede::gateway::{PlatformAdapter, Platform, InboundMessage};
use tokio::sync::mpsc;

#[derive(Clone)]
struct MockAdapter {
    platform: Platform,
    authorized_users: Vec<String>,
}

#[async_trait::async_trait]
impl PlatformAdapter for MockAdapter {
    fn platform_name(&self) -> Platform {
        self.platform.clone()
    }

    async fn start_listening(
        &self,
        inbox_tx: mpsc::UnboundedSender<InboundMessage>,
    ) -> Result<(), polymede::gateway::GatewayError> {
        let msg = InboundMessage {
            platform: self.platform.clone(),
            user_id: "test_user".into(),
            content: "Hello from mock adapter".into(),
            thread_id: None,
            timestamp: chrono::Utc::now(),
        };
        let _ = inbox_tx.send(msg);
        Ok(())
    }

    async fn send_message(&self, _user_id: &str, _content: &str) -> Result<(), polymede::gateway::GatewayError> {
        Ok(())
    }

    async fn shutdown(&self) {}

    fn is_authorized(&self, user_id: &str) -> bool {
        self.authorized_users.iter().any(|u| u == user_id)
    }
}

#[tokio::test]
async fn test_mock_adapter_platform_name() {
    let adapter = MockAdapter {
        platform: Platform::Telegram,
        authorized_users: vec!["user1".into()],
    };
    assert_eq!(adapter.platform_name(), Platform::Telegram);
}

#[tokio::test]
async fn test_mock_adapter_authorization() {
    let adapter = MockAdapter {
        platform: Platform::Discord,
        authorized_users: vec!["alice".into(), "bob".into()],
    };
    assert!(adapter.is_authorized("alice"));
    assert!(adapter.is_authorized("bob"));
    assert!(!adapter.is_authorized("eve"));
}

#[tokio::test]
async fn test_mock_adapter_send_message() {
    let adapter = MockAdapter {
        platform: Platform::Slack,
        authorized_users: vec!["user1".into()],
    };
    assert!(adapter.send_message("user1", "Test message").await.is_ok());
}

#[tokio::test]
async fn test_mock_adapter_shutdown() {
    let adapter = MockAdapter {
        platform: Platform::WhatsApp,
        authorized_users: vec!["user1".into()],
    };
    adapter.shutdown().await;
}

#[tokio::test]
async fn test_inbound_message_structure() {
    let msg = InboundMessage {
        platform: Platform::Signal,
        user_id: "signal_user_123".into(),
        content: "Test signal message".into(),
        thread_id: None,
            timestamp: chrono::Utc::now(),
    };
    assert_eq!(msg.platform, Platform::Signal);
    assert_eq!(msg.user_id, "signal_user_123");
    assert_eq!(msg.content, "Test signal message");
}

#[tokio::test]
async fn test_inbound_message_clone() {
    let msg = InboundMessage {
        platform: Platform::Email,
        user_id: "email@example.com".into(),
        content: "Subject: Test\nBody: Hello".into(),
        thread_id: None,
            timestamp: chrono::Utc::now(),
    };
    let cloned = msg.clone();
    assert_eq!(cloned.platform, msg.platform);
    assert_eq!(cloned.user_id, msg.user_id);
}

#[tokio::test]
async fn test_multiple_adapters_independent() {
    let telegram = MockAdapter {
        platform: Platform::Telegram,
        authorized_users: vec!["tg_user".into()],
    };
    let discord = MockAdapter {
        platform: Platform::Discord,
        authorized_users: vec!["dc_user".into()],
    };
    assert!(telegram.is_authorized("tg_user"));
    assert!(!telegram.is_authorized("dc_user"));
    assert!(discord.is_authorized("dc_user"));
    assert!(!discord.is_authorized("tg_user"));
}

#[tokio::test]
async fn test_adapter_listening_sends_to_inbox() {
    let adapter = MockAdapter {
        platform: Platform::Telegram,
        authorized_users: vec!["test_user".into()],
    };
    let (tx, mut rx) = mpsc::unbounded_channel();

    let handle = tokio::spawn(async move {
        adapter.start_listening(tx).await.unwrap();
    });

    let msg = tokio::time::timeout(std::time::Duration::from_secs(2), rx.recv())
        .await
        .expect("timed out waiting for message")
        .expect("channel closed");

    assert_eq!(msg.platform, Platform::Telegram);
    assert_eq!(msg.user_id, "test_user");
    assert_eq!(msg.content, "Hello from mock adapter");

    handle.await.unwrap();
}
