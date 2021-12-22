# Aegis face detector node

## Usage
POST
```js
"<base64 encoded image>"
```
to `/` to get a list of bounds and confidences like so:
```js
[
  {
    bounds: [x, y, w, h],
    confidence: 0.99
  }
  ...
]
```

## Environment
- `PORT` - the port to listen on
- `MIN_CONFIDENCE` - minimum confidence to accept

## TODO
- config env vars for blobFromImage params
