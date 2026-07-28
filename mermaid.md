```mermaid
graph TD
    A[Raw JSON Email Files] --> B[Ingestion & PII Redaction Pipeline]
    B --> C[(Relational DB: e.g., PostgreSQL)]
    B --> D[(Vector DB: e.g., Qdrant / PgVector)]
    
    subgraph Relational DB (SQL)
        C1[threads Table] -->|1-to-Many| C2[emails Table]
        C2 -->|Stores| C3[From, To, Date, Subject, Redacted Body, ThreadID, ThreadPosition]
    end
    
    subgraph Vector DB (NoSQL/Vector)
        D1[Vector Indexes] -->|Stores| D2[Text Embeddings of Body Chunk]
        D1 -->|Payload Metadata| D3[Subject, From, To, Date, ThreadID, MessageID]
    end


```