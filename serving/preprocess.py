from __future__ import annotations

class Preprocess:
    """Text JSON adapter for ClearML Serving.

    ClearML Serving loads this file as a preprocessing hook. The exact hook
    signature can differ slightly between serving versions, so the methods use
    permissive arguments while keeping the data contract simple:
    request JSON {"text": "..."} -> sklearn input ["..."] -> {"label": "..."}.
    """

    def preprocess(self, body, *args, **kwargs):
        if isinstance(body, dict):
            text = body.get("text") or body.get("data") or body.get("inputs")
        else:
            text = body
        if isinstance(text, list):
            return [str(item) for item in text]
        if text is None:
            raise ValueError("Request body must include a 'text' field")
        return [str(text)]

    def postprocess(self, data, *args, **kwargs):
        if hasattr(data, "tolist"):
            data = data.tolist()
        if isinstance(data, (list, tuple)):
            label = data[0] if data else None
        else:
            label = data
        return {"label": str(label)}

