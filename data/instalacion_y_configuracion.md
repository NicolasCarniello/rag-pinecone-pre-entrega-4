---
category: guia
tags: [instalacion, configuracion, cliente]
---

# andespay-python — Instalación y configuración del cliente

> Documentación ficticia de una librería de Python, creada como dataset de ejemplo para el sistema RAG.
> Versión documentada: andespay 2.4.

## Requisitos

La librería `andespay` requiere **Python 3.10 o superior**. No tiene dependencias nativas: usa `httpx` para las llamadas HTTP y `pydantic` v2 para validar las respuestas de la API.

## Instalación

```bash
pip install andespay
```

Para usar el cliente asíncrono hay que instalar el extra `async`, que agrega soporte para HTTP/2:

```bash
pip install "andespay[async]"
```

## Crear el cliente

El punto de entrada es la clase `AndesPayClient`:

```python
from andespay import AndesPayClient

client = AndesPayClient(api_key="sk_test_123", environment="sandbox")
```

Parámetros principales del constructor:

| Parámetro | Tipo | Por defecto | Descripción |
|-----------|------|-------------|-------------|
| `api_key` | `str` | `None` | Clave secreta. Si es `None`, se lee de la variable de entorno `ANDESPAY_API_KEY`. |
| `environment` | `str` | `"sandbox"` | `"sandbox"` o `"production"`. |
| `timeout` | `float` | `30.0` | Segundos de espera máxima por petición. |
| `max_retries` | `int` | `2` | Reintentos automáticos ante errores de red y respuestas 5xx o 429. |
| `api_version` | `str` | `"2026-01"` | Versión de la API que se envía en el encabezado `AndesPay-Version`. |

Si no se pasa `api_key` y la variable `ANDESPAY_API_KEY` tampoco está definida, el constructor lanza `AuthenticationError` con el código `AP-1001`.

Las claves de sandbox empiezan con `sk_test_` y las de producción con `sk_live_`. Si se usa una clave `sk_live_` con `environment="sandbox"` (o al revés), el cliente lanza `EnvironmentMismatchError` antes de hacer cualquier llamada.

## Cliente asíncrono

Para aplicaciones con `asyncio` (FastAPI, por ejemplo) existe `AsyncAndesPayClient`, que tiene exactamente los mismos métodos pero como corrutinas:

```python
from andespay import AsyncAndesPayClient

async with AsyncAndesPayClient() as client:
    payment = await client.payments.retrieve("pay_9f2c")
```

Se recomienda usarlo como *context manager* para que las conexiones se cierren al terminar. Si se crea sin `async with`, hay que llamar a `await client.aclose()`.

## Reintentos automáticos

El cliente reintenta automáticamente las peticiones que fallan por errores de red, por `429 Too Many Requests` o por errores 5xx. Usa *backoff* exponencial con *jitter*, empezando en 0,5 segundos y con un máximo de 8 segundos entre intentos. Si la respuesta incluye el encabezado `Retry-After`, el cliente respeta ese valor.

Las peticiones `POST` solo se reintentan si tienen `idempotency_key`; si no la tienen, el SDK genera una automáticamente (un UUID v4) para que el reintento sea seguro.

Para desactivar los reintentos en una llamada puntual se usa `client.with_options(max_retries=0)`.

## Logging

La librería usa el logger estándar `andespay`. Para ver cada petición y su respuesta:

```python
import logging
logging.getLogger("andespay").setLevel(logging.DEBUG)
```

También se puede activar con la variable de entorno `ANDESPAY_LOG=debug`. Los números de tarjeta y las API keys se enmascaran siempre en los logs.

## Proxy y certificados

El cliente respeta las variables `HTTPS_PROXY` y `SSL_CERT_FILE`. Para un proxy específico se puede pasar un `httpx.Client` propio con el parámetro `http_client`.
