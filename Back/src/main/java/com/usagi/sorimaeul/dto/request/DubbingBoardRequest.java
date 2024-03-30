package com.usagi.sorimaeul.dto.request;

import lombok.Getter;

@Getter
public class DubbingBoardRequest {

    private String dubName;
    private String dubDetail;
    private int videoSourceCode;
    private Boolean isPublic;
}
