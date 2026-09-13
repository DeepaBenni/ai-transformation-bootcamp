CREATE DATABASE IF NOT EXISTS opsmate
  CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE opsmate;

-- ---------------------------------------------------------------- estate ----

CREATE TABLE IF NOT EXISTS users (
  user_id           VARCHAR(16)  NOT NULL PRIMARY KEY,
  full_name         VARCHAR(128) NOT NULL,
  email             VARCHAR(128) NOT NULL,
  department        VARCHAR(64)  NOT NULL,
  manager           VARCHAR(128) NULL,
  employment_status ENUM('active','leaver','contractor','suspended') NOT NULL DEFAULT 'active',
  created_at        DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS accounts (
  user_id           VARCHAR(16) NOT NULL PRIMARY KEY,
  state             ENUM('active','locked','disabled') NOT NULL DEFAULT 'active',
  failed_logins_24h INT         NOT NULL DEFAULT 0,
  disabled_reason   VARCHAR(128) NULL,
  last_login_at     DATETIME     NULL,
  password_set_at   DATETIME     NULL,
  CONSTRAINT fk_accounts_user FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS hosts (
  host        VARCHAR(64) NOT NULL PRIMARY KEY,
  os          VARCHAR(64) NOT NULL,
  environment ENUM('prod','staging','dev') NOT NULL DEFAULT 'prod',
  site        VARCHAR(64) NOT NULL
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS services (
  service        VARCHAR(64) NOT NULL PRIMARY KEY,
  host           VARCHAR(64) NOT NULL,
  status         ENUM('up','degraded','down','restarting') NOT NULL DEFAULT 'up',
  depends_on     VARCHAR(128) NULL,
  status_since   DATETIME    NULL,
  restart_window VARCHAR(32) NOT NULL DEFAULT '22:00-04:00',
  CONSTRAINT fk_services_host FOREIGN KEY (host) REFERENCES hosts(host) ON DELETE CASCADE
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS config_items (
  ci_name        VARCHAR(64) NOT NULL PRIMARY KEY,
  ci_type        VARCHAR(32) NOT NULL,
  status         ENUM('healthy','degraded','failed') NOT NULL DEFAULT 'healthy',
  cpu_pct        DECIMAL(5,2) NOT NULL DEFAULT 0,
  mem_pct        DECIMAL(5,2) NOT NULL DEFAULT 0,
  last_checkin_at DATETIME    NULL,
  owner_team     VARCHAR(64) NOT NULL
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS disk_usage (
  id       INT AUTO_INCREMENT PRIMARY KEY,
  host     VARCHAR(64) NOT NULL,
  mount    VARCHAR(64) NOT NULL,
  used_pct DECIMAL(5,2) NOT NULL,
  free_gb  DECIMAL(8,2) NOT NULL,
  UNIQUE KEY uq_disk_host_mount (host, mount),
  CONSTRAINT fk_disk_host FOREIGN KEY (host) REFERENCES hosts(host) ON DELETE CASCADE
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS access_groups (
  group_name       VARCHAR(64)  NOT NULL PRIMARY KEY,
  description      VARCHAR(255) NOT NULL,
  owner            VARCHAR(128) NOT NULL,
  manager_approval TINYINT(1)   NOT NULL DEFAULT 1
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS group_memberships (
  id         INT AUTO_INCREMENT PRIMARY KEY,
  user_id    VARCHAR(16) NOT NULL,
  group_name VARCHAR(64) NOT NULL,
  granted_at DATETIME    NOT NULL DEFAULT CURRENT_TIMESTAMP,
  UNIQUE KEY uq_membership (user_id, group_name),
  CONSTRAINT fk_gm_user  FOREIGN KEY (user_id)    REFERENCES users(user_id)          ON DELETE CASCADE,
  CONSTRAINT fk_gm_group FOREIGN KEY (group_name) REFERENCES access_groups(group_name) ON DELETE CASCADE
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS tickets (
  ticket_id  VARCHAR(24) NOT NULL PRIMARY KEY,
  user_id    VARCHAR(16) NOT NULL,
  body       TEXT        NOT NULL,
  category   VARCHAR(32) NOT NULL,
  status     ENUM('open','resolved','escalated') NOT NULL DEFAULT 'resolved',
  resolution VARCHAR(512) NULL,
  created_at DATETIME    NOT NULL,
  closed_at  DATETIME    NULL,
  KEY idx_tickets_user (user_id),
  CONSTRAINT fk_tickets_user FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
) ENGINE=InnoDB;

-- --------------------------------------------------------------- runtime ----

CREATE TABLE IF NOT EXISTS runs (
  run_id       VARCHAR(40) NOT NULL PRIMARY KEY,
  ticket_id    VARCHAR(24) NULL,
  ticket_body  TEXT        NOT NULL,
  path         ENUM('undecided','resolve','clarify','escalate') NOT NULL DEFAULT 'undecided',
  status       ENUM('running','awaiting_approval','awaiting_reply','done','error')
               NOT NULL DEFAULT 'running',
  stop_reason  VARCHAR(512) NULL,
  total_tokens INT          NOT NULL DEFAULT 0,
  cost_usd     DECIMAL(10,6) NOT NULL DEFAULT 0,
  latency_ms   INT          NOT NULL DEFAULT 0,
  created_at   DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
  ended_at     DATETIME     NULL
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS approvals (
  id         INT AUTO_INCREMENT PRIMARY KEY,
  run_id     VARCHAR(40)  NOT NULL,
  token_hash CHAR(64)     NOT NULL,
  action     VARCHAR(64)  NOT NULL,
  args_json  JSON         NOT NULL,
  decision   ENUM('approved','denied') NOT NULL,
  approver   VARCHAR(128) NOT NULL,
  reason     TEXT         NULL,
  decided_at DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
  UNIQUE KEY uq_approval_token (token_hash),
  KEY idx_approvals_run (run_id)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS audit_events (
  id                INT AUTO_INCREMENT PRIMARY KEY,
  run_id            VARCHAR(40) NOT NULL,
  seq               INT         NOT NULL,
  ts                DATETIME(3) NOT NULL,
  agent             VARCHAR(32) NULL,
  event             VARCHAR(32) NOT NULL,
  tool              VARCHAR(64) NULL,
  args_json         JSON        NULL,
  observation_json  JSON        NULL,
  model             VARCHAR(64) NULL,
  prompt_tokens     INT         NOT NULL DEFAULT 0,
  completion_tokens INT         NOT NULL DEFAULT 0,
  total_tokens      INT         NOT NULL DEFAULT 0,
  cost_usd          DECIMAL(10,6) NOT NULL DEFAULT 0,
  latency_ms        INT         NOT NULL DEFAULT 0,
  error             TEXT        NULL,
  UNIQUE KEY uq_audit_run_seq (run_id, seq),
  KEY idx_audit_run (run_id)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS action_log (
  id          INT AUTO_INCREMENT PRIMARY KEY,
  run_id      VARCHAR(40)  NOT NULL,
  action      VARCHAR(64)  NOT NULL,
  args_json   JSON         NOT NULL,
  approver    VARCHAR(128) NOT NULL,
  result      VARCHAR(512) NOT NULL,
  executed_at DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
  KEY idx_action_run (run_id)
) ENGINE=InnoDB;

-- ---------------------------------------------------------------- chat ----
-- The Streamlit / CLI conversation history: one row per session, and one row
-- per turn. This is what the "session history" side panel lists and reloads.

CREATE TABLE IF NOT EXISTS chat_sessions (
  session_id VARCHAR(64)  NOT NULL PRIMARY KEY,
  seq        INT          NOT NULL,
  title      VARCHAR(200) NOT NULL,
  facts      JSON         NULL,
  created_at DATETIME(3)  NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
  updated_at DATETIME(3)  NOT NULL DEFAULT CURRENT_TIMESTAMP(3) ON UPDATE CURRENT_TIMESTAMP(3),
  UNIQUE KEY uq_chat_seq (seq)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS chat_messages (
  id         INT AUTO_INCREMENT PRIMARY KEY,
  session_id VARCHAR(64) NOT NULL,
  seq        INT         NOT NULL,
  role       ENUM('user','assistant') NOT NULL,
  kind       VARCHAR(16) NOT NULL DEFAULT 'text',   -- text | run | chat
  content    TEXT        NOT NULL,
  run_id     VARCHAR(40) NULL,
  payload    JSON        NULL,                       -- the full API response for a re-render
  created_at DATETIME(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
  UNIQUE KEY uq_msg_session_seq (session_id, seq),
  KEY idx_msg_session (session_id),
  CONSTRAINT fk_msg_session FOREIGN KEY (session_id)
    REFERENCES chat_sessions(session_id) ON DELETE CASCADE
) ENGINE=InnoDB;
