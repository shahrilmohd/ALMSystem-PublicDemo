"""
knowledge_base — context injection sources for ALM agent system prompts.

Four sources assembled at agent instantiation:
  schema_export.py       — live RunConfig JSON schema (auto-generated)
  decisions_loader.py    — relevant DECISIONS.md sections
  architecture_loader.py — relevant ALM_Architecture.md sections
  model_docs.md          — hand-curated actuarial glossary
"""
