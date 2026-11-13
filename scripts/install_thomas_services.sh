#!/bin/zsh
set -euo pipefail

ROOT="/Users/local.local/Projects/TradingSystem"
AGENTS="$HOME/Library/LaunchAgents"
UID_VALUE="$(id -u)"
mkdir -p "$AGENTS" "$ROOT/logs"

cat > "$AGENTS/com.thomas-ai.mlx.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.thomas-ai.mlx</string>
  <key>ProgramArguments</key><array><string>/bin/zsh</string><string>$ROOT/scripts/start_thomas_mlx.sh</string></array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>ThrottleInterval</key><integer>10</integer>
  <key>ProcessType</key><string>Interactive</string>
  <key>StandardOutPath</key><string>$ROOT/logs/thomas-mlx.log</string>
  <key>StandardErrorPath</key><string>$ROOT/logs/thomas-mlx.err</string>
</dict></plist>
PLIST

cat > "$AGENTS/com.thomas-ai.gateway.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.thomas-ai.gateway</string>
  <key>ProgramArguments</key><array><string>/bin/zsh</string><string>$ROOT/scripts/start_thomas_ai.sh</string></array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>ThrottleInterval</key><integer>10</integer>
  <key>ProcessType</key><string>Interactive</string>
  <key>StandardOutPath</key><string>$ROOT/logs/thomas-ai.log</string>
  <key>StandardErrorPath</key><string>$ROOT/logs/thomas-ai.err</string>
</dict></plist>
PLIST

plutil -lint "$AGENTS/com.thomas-ai.mlx.plist" "$AGENTS/com.thomas-ai.gateway.plist"
launchctl bootout "gui/$UID_VALUE/com.thomas-ai.gateway" 2>/dev/null || true
launchctl bootout "gui/$UID_VALUE/com.thomas-ai.mlx" 2>/dev/null || true
launchctl bootstrap "gui/$UID_VALUE" "$AGENTS/com.thomas-ai.mlx.plist"
launchctl bootstrap "gui/$UID_VALUE" "$AGENTS/com.thomas-ai.gateway.plist"
launchctl enable "gui/$UID_VALUE/com.thomas-ai.mlx"
launchctl enable "gui/$UID_VALUE/com.thomas-ai.gateway"
echo "Thomas AI services installed and loaded"
