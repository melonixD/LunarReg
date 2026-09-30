export default function handler(_request, response) {
  const connected = Boolean(process.env.INFERENCE_API_URL);
  response.status(200).json({
    connected,
    mode: connected ? "remote-inference" : "browser-baseline",
    registrationModel: "not-trained",
    browserEngine: "phase-correlation translation estimator",
    craterModel: connected ? "available-from-inference-service" : "download-only",
    note: connected
      ? "Requests can be proxied to the configured inference service."
      : "No external inference API is configured. Registration runs locally in the browser."
  });
}

