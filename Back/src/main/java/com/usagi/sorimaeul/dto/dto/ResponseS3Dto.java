package com.usagi.sorimaeul.dto.dto;

import lombok.Builder;
import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
@Builder
public class ResponseS3Dto {
    /* S3 Presigned URL 주소 */
    private String url;
}