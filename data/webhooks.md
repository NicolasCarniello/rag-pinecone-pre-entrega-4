---
category: referencia-api
tags: [webhooks, firma, seguridad, eventos]
---

# andespay-python — Webhooks y verificación de firma

> Documentación ficticia de una librería de Python, creada como dataset de ejemplo para el sistema RAG.

## Qué son

Los webhooks avisan a tu servidor cuando ocurre algo en tu cuenta: un pago aprobado (`payment.succeeded`), un reembolso (`refund.created`), una disputa (`dispute.opened`), etc. Andes Pay hace un `POST` con un JSON a la URL que configures en el panel.

## Verificar la firma

Cada webhook trae el encabezado `AndesPay-Signature` con el formato:

```
t=1767225600,v1=5257a869e7ecebeda32affa62cdca3fa51cad7e77a0e56ff536d0ce8e108d8bd
```

- `t` es el timestamp Unix del envío.
- `v1` es un **HMAC-SHA256** de la cadena `"{t}.{cuerpo_crudo}"` calculado con el *signing secret* del endpoint (empieza con `whsec_`).

El SDK lo verifica con `WebhookVerifier`:

```python
from andespay import WebhookVerifier, SignatureVerificationError

verifier = WebhookVerifier(secret="whsec_abc123", tolerance=300)

try:
    event = verifier.construct_event(payload=request.body, sig_header=request.headers["AndesPay-Signature"])
except SignatureVerificationError:
    return 400
```

Puntos importantes:

- `payload` tiene que ser el **cuerpo crudo en bytes**, tal cual llegó. Si el framework ya lo parseó a JSON y lo volvés a serializar, la firma no coincide.
- `tolerance` es la antigüedad máxima aceptada en segundos. El valor por defecto es **300 segundos (5 minutos)**; protege contra ataques de repetición (*replay*).
- Si la firma no coincide o el timestamp es demasiado viejo, se lanza `SignatureVerificationError` (código `AP-3002`).
- La comparación se hace en tiempo constante (`hmac.compare_digest`).

## Rotación del signing secret

Al rotar el secret desde el panel, el anterior sigue siendo válido durante 24 horas. Durante ese período el encabezado trae dos firmas `v1`, una por cada secret, y `WebhookVerifier` acepta una lista: `WebhookVerifier(secret=["whsec_nuevo", "whsec_viejo"])`.

## Reintentos y orden

Tu endpoint tiene que responder `2xx` en menos de **10 segundos**. Si no, Andes Pay reintenta hasta **8 veces durante 3 días** con *backoff* exponencial.

Los eventos pueden llegar **duplicados o desordenados**. Para procesarlos una sola vez, guardá `event.id` y descartá los que ya viste. Para el orden, usá `event.created` o volvé a consultar el objeto con `client.payments.retrieve(...)` en lugar de confiar en el contenido del evento.

## Probar webhooks en local

El CLI incluido en el paquete reenvía eventos de sandbox a tu máquina:

```bash
andespay listen --forward-to localhost:8000/webhooks
```

El comando imprime un signing secret temporal (`whsec_local_...`) que hay que usar en `WebhookVerifier` mientras se prueba. Con `andespay trigger payment.succeeded` se dispara un evento de prueba.

## Tipos de evento más usados

| Evento | Cuándo se envía |
|--------|-----------------|
| `payment.succeeded` | Un pago se cobró. |
| `payment.failed` | Un pago fue rechazado. |
| `refund.created` | Se creó un reembolso. |
| `dispute.opened` | El comprador desconoció el cargo ante su banco. |
| `payout.paid` | Se acreditó una liquidación en tu cuenta bancaria. |
