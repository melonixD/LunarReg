import { proxyMultipart } from "./_proxy.js";

export const config = { api: { bodyParser: false } };

export default async function handler(request, response) {
  return proxyMultipart(request, response, "/register");
}

