export async function proxyMultipart(request, response, endpoint) {
  if (request.method !== "POST") {
    response.setHeader("Allow", "POST");
    return response.status(405).json({ error: "Method not allowed" });
  }

  const baseUrl = process.env.INFERENCE_API_URL;
  if (!baseUrl) {
    return response.status(503).json({
      error: "Inference service is not configured",
      fallback: "Use the browser baseline or set INFERENCE_API_URL in Vercel."
    });
  }

  const chunks = [];
  let size = 0;
  const maxBytes = 4 * 1024 * 1024;

  for await (const chunk of request) {
    size += chunk.length;
    if (size > maxBytes) {
      return response.status(413).json({ error: "Upload exceeds the 4 MB proxy limit" });
    }
    chunks.push(chunk);
  }

  try {
    const upstream = await fetch(`${baseUrl.replace(/\/$/, "")}${endpoint}`, {
      method: "POST",
      headers: {
        "content-type": request.headers["content-type"] || "application/octet-stream"
      },
      body: Buffer.concat(chunks)
    });
    const contentType = upstream.headers.get("content-type") || "application/json";
    response.status(upstream.status).setHeader("content-type", contentType);
    response.send(Buffer.from(await upstream.arrayBuffer()));
  } catch (error) {
    response.status(502).json({ error: "Inference service unavailable", detail: error.message });
  }
}

