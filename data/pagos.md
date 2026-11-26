---
category: referencia-api
tags: [pagos, payments, idempotencia, estados]
---

# andespay-python — Referencia: Pagos (`client.payments`)

> Documentación ficticia de una librería de Python, creada como dataset de ejemplo para el sistema RAG.

## Crear un pago

```python
payment = client.payments.create(
    amount=150000,            # en centavos: 1500,00
    currency="ARS",
    payment_method="pm_card_visa",
    description="Pedido #1042",
    capture=True,
    idempotency_key="pedido-1042",
)
```

- `amount` es un entero en la **unidad mínima** de la moneda (centavos). Pasar un `float` lanza `ValidationError` con el código `AP-2003`.
- `currency` acepta códigos ISO 4217: `ARS`, `USD`, `BRL`, `CLP`, `UYU`. Para `CLP` el monto no lleva decimales.
- El monto mínimo es el equivalente a 1 USD; por debajo, la API responde `AP-2010 AmountTooSmallError`.

## Autorizar y capturar por separado

Con `capture=False` el pago solo se **autoriza**: el dinero queda reservado en la tarjeta del comprador pero no se cobra. Después hay que capturarlo:

```python
client.payments.capture("pay_9f2c", amount=120000)  # captura parcial permitida
```

Una autorización vence a los **7 días**. Si no se captura antes, pasa al estado `expired` y el dinero se libera. Se puede capturar una sola vez; para capturar un monto menor al autorizado se indica `amount` y la diferencia se libera automáticamente.

Para cancelar una autorización sin capturarla se usa `client.payments.void("pay_9f2c")`.

## Estados de un pago

El campo `payment.status` es un valor del enum `PaymentStatus`:

| Estado | Significado |
|--------|-------------|
| `PaymentStatus.PENDING` | Creado, esperando confirmación del medio de pago (por ejemplo, transferencia). |
| `PaymentStatus.AUTHORIZED` | Autorizado, pendiente de captura. |
| `PaymentStatus.SUCCEEDED` | Cobrado. |
| `PaymentStatus.FAILED` | Rechazado. El motivo está en `payment.failure_code`. |
| `PaymentStatus.EXPIRED` | Autorización vencida sin capturar. |
| `PaymentStatus.REFUNDED` | Reembolsado total o parcialmente. |

Las transiciones válidas son: `PENDING → SUCCEEDED | FAILED`, `AUTHORIZED → SUCCEEDED | EXPIRED`, y `SUCCEEDED → REFUNDED`.

## Idempotencia

Toda llamada a `create` acepta `idempotency_key`. La API guarda la respuesta asociada a esa clave durante **24 horas**:

- Misma clave y mismo cuerpo: devuelve el pago original, sin cobrar de nuevo.
- Misma clave y cuerpo distinto: el SDK lanza `IdempotencyConflictError` (código `AP-4091`, HTTP 409).

Se recomienda derivar la clave del identificador interno del pedido (por ejemplo `pedido-1042`) y no de un valor aleatorio, para que los reintentos de tu propio sistema también sean seguros.

## Consultar y listar pagos

```python
payment = client.payments.retrieve("pay_9f2c")

for p in client.payments.list(status="succeeded", created_gte="2026-01-01"):
    print(p.id, p.amount)
```

`list` devuelve un iterador que pagina automáticamente (ver la guía de paginación). Los filtros disponibles son `status`, `currency`, `customer` y rangos de fecha con `created_gte` y `created_lt`.

## Metadatos propios

Cada pago acepta un diccionario `metadata` con hasta **20 claves**; cada clave hasta 40 caracteres y cada valor hasta 500. Sirve para guardar referencias propias (ID de pedido, canal de venta) y después buscar por ellas con `client.payments.search(query="metadata['pedido']:'1042'")`.
