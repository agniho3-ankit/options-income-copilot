"""Local web API for the review cockpit (UI phase).

A thin layer over the existing `options_copilot` functions. It holds no trading logic: every
number, verdict, score, tier and signal comes from the Python modules it calls. It binds to
127.0.0.1 only, checks a per-session CSRF token on every state-changing request, and serves
the built front end from `ui/dist` with no third-party assets.
"""
