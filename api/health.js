export default function handler(_request, response) {
  response.status(200).json({
    status: "ok",
    service: "LunarReg web",
    version: "1.0.0",
    timestamp: new Date().toISOString()
  });
}

