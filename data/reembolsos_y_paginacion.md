---
category: referencia-api
tags: [reembolsos, refunds, paginacion, cursor]
---

# andespay-python — Reembolsos y paginación

> Documentación ficticia de una librería de Python, creada como dataset de ejemplo para el sistema RAG.

## Crear un reembolso

```python
refund = client.refunds.create(payment="pay_9f2c", amount=50000, reason="requested_by_customer")
```

- Si se omite `amount`, el reembolso es **total**.
- Se permiten hasta **5 reembolsos parciales** por pago, siempre que la suma no supere el monto cobrado.
- `reason` acepta `requested_by_customer`, `duplicate` o `fraudulent`. Marcar `fraudulent` además agrega la tarjeta a la lista de bloqueo de tu cuenta.
- Solo se puede reembolsar un pago en estado `SUCCEEDED` y dentro de los **180 días** desde el cobro. Fuera de ese plazo, la API responde `AP-2045 RefundWindowExpiredError`.

El estado del reembolso está en `refund.status`: `pending`, `succeeded` o `failed`. Los reembolsos a tarjeta de crédito suelen quedar `pending` entre 1 y 5 días hábiles.

## Paginación por cursor

Todos los métodos `list` de la API (pagos, reembolsos, clientes, eventos) usan **paginación por cursor**, no por número de página. Cada página trae hasta `limit` elementos (por defecto 25, máximo 100) y un campo `next_cursor`.

### Paginación automática (recomendada)

El objeto que devuelve `list` es iterable y pide las páginas siguientes a medida que se recorre:

```python
for refund in client.refunds.list(limit=100):
    procesar(refund)
```

Para cortar después de cierta cantidad de elementos, se usa `itertools.islice` o el parámetro `max_items`.

### Paginación manual

Cuando hace falta guardar la posición (por ejemplo, en un job que procesa por tandas), se usa la página explícita:

```python
page = client.refunds.list(limit=100).first_page()
while True:
    procesar(page.data)
    if not page.has_more:
        break
    page = client.refunds.list(limit=100, cursor=page.next_cursor).first_page()
```

- `page.data` es la lista de elementos de esa página.
- `page.has_more` indica si hay más.
- `page.next_cursor` es un string opaco: no hay que interpretarlo ni construirlo a mano. Un cursor vence a las **24 horas**; usar uno vencido lanza `AP-4100 InvalidCursorError`.

### Orden y consistencia

Los resultados se devuelven del más nuevo al más viejo (`created` descendente). La paginación por cursor es estable: si se crean elementos nuevos mientras se recorre, no se duplican ni se saltean elementos ya listados.

En el cliente asíncrono la iteración automática se hace con `async for`:

```python
async for refund in client.refunds.list(limit=100):
    await procesar(refund)
```

## Diferencia con versiones anteriores

Hasta la versión 2.2 del SDK, `refunds.list` usaba los parámetros `page` y `per_page`. Esos parámetros fueron eliminados en la 2.3 (ver el changelog).
